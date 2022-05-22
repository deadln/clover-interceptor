#!/usr/bin/env python
import rospy
from clover import srv
from std_srvs.srv import Trigger
from std_msgs.msg import String
from geometry_msgs.msg import Point32
from std_msgs.msg import Bool
from cv_bridge import CvBridge

import numpy as np
import math
import random
from enum import Enum


node_name = "CopterController"

class CopterController():
    def __init__(self):
        rospy.init_node(node_name)
        rospy.loginfo(node_name + " started")
        # Сервисы Клевера
        self.__get_telemetry__ = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.__navigate__ = rospy.ServiceProxy('navigate', srv.Navigate)
        self.__set_position__ = rospy.ServiceProxy('set_position', srv.SetPosition)
        self.__set_velocity__ = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
        self.__land__ = rospy.ServiceProxy('land', Trigger)

        self.bridge = CvBridge()
        # Константы, отвечающие за логику работы системы
        self.FREQUENCY = 5
        self.DEPTH_QUEUE_SIZE = 500
        self.CAMERA_ANGLE_H = 1.5009831567151235
        self.CAMERA_ANGLE_V = 0.9948376736367679
        self.CRITICAL_CELL_VOLTAGE = 3.1
        self.SUSPICION_DURATION = 5
        self.PURSUIT_DURATION = 2
        self.SEARCH_DURATION = 7
        self.SUSPICION_TRIGGER_COUNT = 10
        self.PURSUIT_TRIGGER_COUNT = 15

        # Константы, связанные с полётом
        self.X_NORM = np.array([1, 0, 0])
        self.SPIN_TIME = 8
        self.SPIN_RATE = 2 * math.pi / self.SPIN_TIME
        self.PATROL_SPEED = 0.3
        self.INTERCEPTION_SPEED = 0.5
        self.DETECTION_DIAPASON_SEC = 1.0
        self.PATROL_TARGET_BORDER_BIAS = 0.5

        # TODO: парсить данные о полётной зоне из txt или launch файла
        # Переменные системы
        # Зона патрулирования и запретная зона
        self.low_left_corner = np.array([0.5, 0.5, 0.5])
        self.up_right_corner = np.array([7.0, 4.8, 3.2])
        self.low_left_corner_restricted = np.array([0, 0, 0])
        self.up_right_corner_restricted = np.array([2.0, 2.0, 3.2])
        self.base = None
        self.telemetry = None
        self.state = ""
        self.state_timestamp = rospy.get_time()
        self.patrol_target = None
        self.spin_start = None
        self.consecutive_detections = 0
        self.suspicion_target = None
        self.pursuit_target = None
        self.detection_pixel = Point32(0, 0, 0)
        self.pursuit_target_detections = []
        self.depth_images = []
        # Создание топиков и подписка на них
        self.telemetry_pub = rospy.Publisher("/telemetry_topic", String, queue_size=10)
        rospy.Subscriber("drone_detection/target_position", String, self.target_callback)
        rospy.Subscriber("load_cell/catch", Bool, self.catch_callback)

        rospy.on_shutdown(self.on_shutdown_cb)

    # Обёртки для сервисов Клевера, принимающие вектора Numpy
    def get_telemetry(self, frame_id='aruco_map'):
        if frame_id == 'aruco_map':
            return self.telemetry
        telemetry = self.__get_telemetry__(frame_id=frame_id)
        return telemetry

    def get_position(self, frame_id='aruco_map'):
        if frame_id == 'aruco_map':
            return np.array([self.telemetry.x, self.telemetry.y, self.telemetry.z])
        telemetry = self.__get_telemetry__(frame_id=frame_id)
        return np.array([telemetry.x, telemetry.y, telemetry.z])

    def navigate(self, target=np.array([0, 0, 2]), speed=0.5, yaw=float('nan'), yaw_rate=0.0, auto_arm=False, frame_id='aruco_map'):
        self.__navigate__(x=target[0], y=target[1], z=target[2], speed=speed, yaw=yaw, yaw_rate=yaw_rate, auto_arm=auto_arm, frame_id=frame_id)

    def set_position(self, target=np.array([0, 0, 2]), speed=0.5, yaw=float('nan'), yaw_rate=0.0, auto_arm=False, frame_id='aruco_map'):
        self.__set_position__(x=target[0], y=target[1], z=target[2], speed=speed, yaw=yaw, yaw_rate=yaw_rate, auto_arm=auto_arm, frame_id=frame_id)

    def set_velocity(self, target=np.array([0, 0, 0]), yaw=float('nan'), yaw_rate=0.0, auto_arm=False, frame_id='aruco_map'):
        self.__set_velocity__(vx=target[0], vy=target[1], vz=target[2], yaw=yaw, yaw_rate=yaw_rate, auto_arm=auto_arm, frame_id=frame_id)

    def land(self):
        return self.__land__()

    # Главный цикл программы
    def offboard_loop(self):
        self.takeoff()
        self.telemetry = self.__get_telemetry__(frame_id='aruco_map')
        self.base = self.get_position() + np.array([0, 0, 0.2])

        rate = rospy.Rate(self.FREQUENCY)
        while not rospy.is_shutdown():
            self.telemetry = self.__get_telemetry__(frame_id='aruco_map')
            self.telemetry_pub.publish(
                f"{self.telemetry.x} {self.telemetry.y} {self.telemetry.z} {self.telemetry.roll} {self.telemetry.pitch} {self.telemetry.yaw}")
            if self.telemetry.cell_voltage < self.CRITICAL_CELL_VOLTAGE:  # Экстренная посадка при разряженной батарее
                rospy.logfatal("CRITICAL CELL VOLTAGE: {}".format(self.telemetry.cell_voltage))
                rospy.signal_shutdown("Cell voltage is too low")
            self.check_state_duration()
            if not self.is_inside_patrol_zone():
                self.return_to_patrol_zone()
                continue
            if self.state == State.PATROL_NAVIGATE:  # Полёт к точке патрулирования
                if self.patrol_target is None:  # Назначение точки патрулирования
                    self.set_patrol_target()
                    rospy.loginfo(f"New patrol target {self.patrol_target}")
                else:
                    # Полёт напрямую
                    self.navigate(self.patrol_target, speed=self.PATROL_SPEED, yaw=self.get_yaw_angle(self.X_NORM, self.patrol_target - self.get_position()))
                if self.is_navigate_target_reached(target=self.patrol_target):  # Argument: target=self.patrol_target
                    rospy.loginfo("Patrol target reached")
                    self.patrol_target = None
            if self.state == State.PURSUIT:  # Состояние преследования цели, которая однозначно обнаружена
                if self.pursuit_target is None:
                    self.set_state(State.PATROL_NAVIGATE)
                else:
                    position = self.get_position(frame_id='aruco_map')
                    error = self.pursuit_target + np.array([0, 0, 0.7]) - position
                    velocity = error / np.linalg.norm(error) * self.INTERCEPTION_SPEED
                    print(f"In pursuit. Interception velocity {velocity}")
                    self.set_velocity(velocity, yaw=self.get_yaw_angle(self.X_NORM, self.pursuit_target - self.get_position()))

            if self.state == State.SUSPICION:  # Проверка места, в котором с т.з. нейросети "мелькнул дрон"
                suspicion_vector = self.suspicion_target - self.get_position()
                suspicion_vector = suspicion_vector * ((np.linalg.norm(suspicion_vector) - 1) / np.linalg.norm(suspicion_vector))
                suspicion_point = self.get_position() + suspicion_vector
                self.navigate(suspicion_point, speed=self.PATROL_SPEED, yaw=self.get_yaw_angle(self.X_NORM, suspicion_point - self.get_position()))

            if self.state == State.SEARCH:
                if self.up_sector(self.detection_pixel.x, self.detection_pixel.y):
                    print("TARGET LOST AT UP SECTOR")
                    if self.left_sector(self.detection_pixel.x, self.detection_pixel.y):
                        yaw_rate = -self.SPIN_RATE
                        print("SPIN LEFT")
                    else:
                        yaw_rate = self.SPIN_RATE
                        print("SPIN RIGHT")

                    self.set_velocity(np.array([0, 0, 0.2]), yaw=float('nan'), yaw_rate=yaw_rate)
                elif self.down_sector(self.detection_pixel.x, self.detection_pixel.y):
                    print("TARGET LOST AT DOWN SECTOR")
                    if self.left_sector(self.detection_pixel.x, self.detection_pixel.y):
                        yaw_rate = self.SPIN_RATE
                        print("SPIN LEFT")
                    else:
                        yaw_rate = -self.SPIN_RATE
                        print("SPIN RIGHT")
                    self.set_velocity(np.array([0, 0, -0.2]), yaw=float('nan'), yaw_rate=yaw_rate)

            if self.state == State.RTB:  # Возвращение на базу
                self.navigate_wait(self.base)
                rospy.signal_shutdown("Mission complete")

            rate.sleep()
    # Взлёт
    def takeoff(self):
        self.set_velocity(np.array([0, 0, 0.2]), yaw=float('nan'), frame_id="body", auto_arm=True)
        rospy.sleep(0.5)
        self.set_state(State.PATROL_NAVIGATE)
        rospy.loginfo("Takeoff complete")
    # Перемещение с ожиданием прилёта в точку
    def navigate_wait(self, target, yaw=float('nan'), speed=0.2, frame_id='aruco_map', auto_arm=False, tolerance=0.3):
        self.navigate(target, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

        while not rospy.is_shutdown():
            if self.telemetry.cell_voltage < self.CRITICAL_CELL_VOLTAGE:
                rospy.logfatal("CRITICAL CELL VOLTAGE: {}".format(self.telemetry.cell_voltage))
                rospy.signal_shutdown("Cell voltage is too low")
            if self.is_navigate_target_reached(tolerance):
                break
            rospy.sleep(0.2)
    # Назначение точки патрулирования
    def set_patrol_target(self):
        llc = self.low_left_corner + np.ones(3) * self.PATROL_TARGET_BORDER_BIAS
        urc = self.up_right_corner - np.ones(3) * self.PATROL_TARGET_BORDER_BIAS
        self.patrol_target = self.get_position()
        while np.linalg.norm(self.patrol_target - self.get_position()) < np.linalg.norm(
                llc - urc) / 3 and not self.is_inside_restricted_zone():
            self.patrol_target = np.array([random.uniform(llc[0], urc[0]),
                                           random.uniform(llc[1], urc[1]),
                                           random.uniform(llc[2], urc[2])])
    # Получить угол рысканья для поворота дрона
    def get_yaw_angle(self, vector_1, vector_2):
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        if vector_2[1] < 0:
            angle *= -1
        return angle
    # Возврат в зону патрулирования
    def return_to_patrol_zone(self):
        position = self.get_position()
        velocity = np.zeros(3)
        velocity += np.array(list(map(int, position < self.low_left_corner)))
        velocity += np.array(list(map(int, position > self.up_right_corner))) * -1
        velocity *= self.INTERCEPTION_SPEED
        self.set_velocity(velocity)
        rospy.logwarn(f"OUT OF PATROL ZONE. RETURN VECTOR {velocity}")
    # Смена состояния
    def set_state(self, state):
        self.state = state
        rospy.loginfo("Changed state to " + state.value)
        if state == State.SUSPICION or state == State.PURSUIT or state == State.SEARCH:
            self.state_timestamp = rospy.get_time()
    # Определение факта достижения цели, заданной через navigate
    def is_navigate_target_reached(self,  tolerance=0.3, target=None):
        if target is None:
            position = self.get_position(frame_id='navigate_target')
        else:
            position = target - self.get_position()
        return np.linalg.norm(position) < tolerance
    # Точка внутри патрульной зоны
    def is_inside_patrol_zone(self):
        position = self.get_position()
        return all(position >= self.low_left_corner) and all(position <= self.up_right_corner)
    # Точка внутри запретной зоны
    def is_inside_restricted_zone(self):
        return all(self.patrol_target >= self.low_left_corner_restricted) and all(self.patrol_target <= self.up_right_corner_restricted)
    # Проверка длительности состояний
    def check_state_duration(self):
        if self.state == State.SUSPICION and rospy.get_time() - self.state_timestamp > self.SUSPICION_DURATION:
            self.set_state(State.PATROL_NAVIGATE)
        elif self.state == State.PURSUIT and rospy.get_time() - self.state_timestamp > self.PURSUIT_DURATION:
            self.set_state(State.SEARCH)
        elif self.state == State.SEARCH and rospy.get_time() - self.state_timestamp > self.SEARCH_DURATION:
            self.set_state(State.PATROL_NAVIGATE)
    # Функции для определения сектора на изображении в котором последний раз была видна цель
    def y1(self, x):
        return int(0.75 * x)

    def y2(self, x):
        return -1 * int(0.75 * x) + 480

    def up_sector(self, x, y):
        return y <= 240
        # return y < self.y1(x) and y < self.y2(x)

    def down_sector(self, x, y):
        return y > 240
        # return y >= self.y1(x) and y >= self.y2(x)

    def right_sector(self, x, y):
        return x > 320
        # return self.y2(x) < y < self.y1(x)

    def left_sector(self, x, y):
        return x <= 320
        # return self.y1(x) <= y <= self.y2(x)
    # Обработка сообщения о цели
    def target_callback(self, message):
        def is_in_net_range():  # Цель находится напротив сетки
            w_radius = 35
            w_left = 320 - w_radius
            w_right = 320 + w_radius
            h_up = 240
            h_down = 480
            return w_left <= self.detection_pixel.x <= w_right and h_up <= self.detection_pixel.y <= h_down

        message = message.data.split()
        position = Point32(float(message[0]), float(message[1]), float(message[2]))
        if math.isnan(position.x):
            if self.consecutive_detections > 0:
                self.consecutive_detections = 0
            if self.state == State.PURSUIT and not is_in_net_range():
                self.set_state(State.SEARCH)
        else:
            self.consecutive_detections += 1
            self.detection_pixel = Point32(int(message[3]), int(message[4]), 0)
            target = np.array([position.x, position.y, position.z])
            if self.state == State.PURSUIT:
                self.pursuit_target = target
                self.state_timestamp = rospy.get_time()
            elif self.consecutive_detections >= self.PURSUIT_TRIGGER_COUNT:
                # self.state = State.PURSUIT
                self.set_state(State.PURSUIT)
                self.pursuit_target = target
            if self.state == State.SUSPICION:
                self.suspicion_target = target
                self.state_timestamp = rospy.get_time()
            elif self.state != State.PURSUIT and self.consecutive_detections >= self.SUSPICION_TRIGGER_COUNT:
                # self.state = State.SUSPICION
                self.set_state(State.SUSPICION)
                self.suspicion_target = target
                rospy.loginfo("Suspicious detect at " + str(self.suspicion_target))

    # Обработка сообщения ложной цели
    def target_callback_test(self, message):
        if message.data == '':
            self.state = 'patrol_navigate'
            self.pursuit_target = None
            self.patrol_target = None
            return
        message = message.data.split()
        self.pursuit_target = np.array(list(map(float, message)))
        self.state = 'pursuit'
    # Обработка сообщения о том что цель была поймана
    def catch_callback(self, message):
        self.set_state(State.RTB)
    # Действия при выключении
    def on_shutdown_cb(self):
        rospy.logwarn("shutdown")
        self.land()
        rospy.loginfo("landing complete")
        exit(0)

# Перечисление состояний системы
class State(Enum):
    PATROL_NAVIGATE = "PATROL_NAVIGATE"
    SUSPICION = "SUSPICION"
    PURSUIT = "PURSUIT"
    SEARCH = "SEARCH"
    RTB = "RTB"


if __name__ == '__main__':
    controller = CopterController()
    try:
        controller.offboard_loop()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()
