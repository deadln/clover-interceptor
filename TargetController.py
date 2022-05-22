import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32

import math
import numpy as np


class TargetController:
    def __init__(self):
        rospy.init_node('TargetController')

        self.telemetry = Telemetry()
        self.bridge = CvBridge()

        self.DEPTH_QUEUE_SIZE = 200
        self.CAMERA_ANGLE_H = 1.5009831567151235
        self.CAMERA_ANGLE_V = 0.9948376736367679

        self.consecutive_detections = 0
        self.depth_images = []
        self.target_detections = []

        rospy.Subscriber("/telemetry_topic", String, self.telemetry_callback)
        rospy.Subscriber('drone_detection/target', String, self.target_callback)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_image_callback)

        # Вывод карты глубины с целеуказателем
        self.depth_debug = rospy.Publisher("debug/depth", Image, queue_size=10)
        # Цель в координатах относительно дрона
        self.target_local_debug = rospy.Publisher("debug/target_position_local", PointCloud, queue_size=10)
        # Цель в координатах относительно пространства
        self.target_global_debug = rospy.Publisher("debug/target_position_global", PointCloud, queue_size=10)
        # Базис пространства относительно дрона
        self.detection_axis_debug = rospy.Publisher("debug/detection_axis", PointCloud, queue_size=10)
        # Точка обнаруженной цели
        self.target_position = rospy.Publisher("drone_detection/target_position", String, queue_size=10)

    def get_telemetry(self):
        return self.telemetry

    # Позиция, получаемая из телеметрии и преобразуемая в numpy.array
    def get_position(self, frame_id='aruco_map'):
        return np.array([self.telemetry.x, self.telemetry.y, self.telemetry.z])
    # Получение телеметрии
    def telemetry_callback(self, message):
        message = message.data.split()
        self.telemetry.x = float(message[0])
        self.telemetry.y = float(message[1])
        self.telemetry.z = float(message[2])
        self.telemetry.roll = float(message[3])
        self.telemetry.pitch = float(message[4])
        self.telemetry.yaw = float(message[5])
    # Получение карты глубин
    def depth_image_callback(self, message):
        def convert_depth_image(ros_image):
            depth_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
            depth_array = np.array(depth_image, dtype=np.float32) * 0.001  # расстояние в метрах 0.001
            return depth_array
        self.depth_images = self.depth_images[:min(len(self.depth_images), self.DEPTH_QUEUE_SIZE)]
        self.depth_images.insert(0, {'timestamp': {'secs': message.header.stamp.secs, 'nsecs': message.header.stamp.nsecs},
                                  'image': convert_depth_image(message)})

    # Определение положения цели
    def target_callback(self, message):
        # Функция рисует целеуказатель на карте глубин
        def draw_cross(img, x, y):
            CROSS_HALF = 2
            CROSS_HALF_LEN = 30
            for i in range(y - CROSS_HALF, y + CROSS_HALF + 1):
                for j in range(x - CROSS_HALF, x + CROSS_HALF + 1):
                    # Up
                    k = 0
                    while i - k >= 0 and k < CROSS_HALF_LEN:
                        img[i - k][j] = 1000
                        k += 1
                    # Down
                    k = 0
                    while i + k < 480 and k < CROSS_HALF_LEN:
                        img[i + k][j] = 1000
                        k += 1
                    # Left
                    k = 0
                    while j - k >= 0 and k < CROSS_HALF_LEN:
                        img[i][j - k] = 1000
                        k += 1
                    # Right
                    k = 0
                    while j + k < 640 and k < CROSS_HALF_LEN:
                        img[i][j + k] = 1000
                        k += 1
            return img

        # Функция нахождения минимальной точки в радиусе на карте глубин
        def get_min_range(img, x, y):
            SEARCH_RADIUS = 40
            x_start = max(x - SEARCH_RADIUS, 0)
            x_end = min(x + SEARCH_RADIUS, 639)
            y_start = max(y - SEARCH_RADIUS, 0)
            y_end = min(y + SEARCH_RADIUS, 479)

            min_distance = 10
            for i in range(y_start, y_end + 1):
                for j in range(x_start, x_end + 1):
                    if img[i][j] > 0 and img[i][j] < min_distance:
                        min_distance = img[i][j]
            return min_distance

        # Функция поворота вектора
        def turn_vector(vect, axis, angle):
            axis = axis / np.linalg.norm(axis)
            rot_matrix = np.array([
                [math.cos(angle) + (1 - math.cos(angle)) * axis[0] ** 2,
                 (1 - math.cos(angle)) * axis[0] * axis[1] - math.sin(angle) * axis[2],
                 (1 - math.cos(angle)) * axis[0] * axis[2] + math.sin(angle) * axis[1]],
                [(1 - math.cos(angle)) * axis[1] * axis[0] + math.sin(angle) * axis[2],
                 math.cos(angle) + (1 - math.cos(angle)) * axis[1] ** 2,
                 (1 - math.cos(angle)) * axis[1] * axis[2] - math.sin(angle) * axis[0]],
                [(1 - math.cos(angle)) * axis[2] * axis[0] - math.sin(angle) * axis[1],
                 (1 - math.cos(angle)) * axis[2] * axis[1] + math.sin(angle) * axis[0],
                 math.cos(angle) + (1 - math.cos(angle)) * axis[2] ** 2]
            ])
            return np.dot(vect, rot_matrix)

        message = message.data.split()
        x_pix = int(message[0])
        y_pix = int(message[1])
        secs = int(message[2])
        nsecs = int(message[3])
        # Если цель не обнаружена
        if x_pix == -1 or y_pix == -1:
            # self.target_position.publish(Point32(float('nan'), float('nan'), float('nan')))
            self.target_position.publish(f"{float('nan')} {float('nan')} {float('nan')} {x_pix} {y_pix}")
            self.target_detections = self.target_detections[:min(len(self.target_detections), self.DEPTH_QUEUE_SIZE)]
            self.target_detections.append(
                {'timestamp': {'secs': secs, 'nsecs': nsecs}, 'position': None})
            if self.consecutive_detections > 0:
                print("CONSECUTIVE DETECTIONS:", self.consecutive_detections)
                self.consecutive_detections = 0
            return
        # Поиск карты глубин, соответствующей кадру сообщения
        i = 0
        while i < len(self.depth_images) and self.depth_images[i]['timestamp']['secs'] > secs:
            i += 1
        if i == len(self.depth_images):
            return
        start = i
        while i < len(self.depth_images) and self.depth_images[i]['timestamp']['secs'] == secs:
            i += 1
        end = i
        if start >= len(self.depth_images) or end >= len(self.depth_images):
            return
        min_i = start
        for i in range(start, end):
            if abs(self.depth_images[i]['timestamp']['nsecs'] - nsecs) < abs(
                    self.depth_images[min_i]['timestamp']['nsecs'] - nsecs):
                min_i = i  # self.depth_images[min_i]['image'] - нужная карта глубины
        # Вычисление локальных координат цели x_local, y_local, z_local
        z_local = get_min_range(self.depth_images[min_i]['image'], x_pix, y_pix)
        self.depth_debug.publish(
            self.bridge.cv2_to_imgmsg(draw_cross(self.depth_images[min_i]['image'] * 100, x_pix, y_pix)))

        w = 2 * math.sqrt(pow(z_local / math.cos(self.CAMERA_ANGLE_H / 2), 2) - pow(z_local, 2))
        h = 2 * math.sqrt(pow(z_local / math.cos(self.CAMERA_ANGLE_V / 2), 2) - pow(z_local, 2))
        x_local = (x_pix - 320) / 640 * w  # (w / 2)
        y_local = -1 * (y_pix - 240) / 480 * h  # (h / 2)
        # Дебаг вывод цели относительно точки (0,0,0)
        point = Point32(x_local, z_local, y_local)
        cloud = PointCloud()
        cloud.header.stamp = rospy.Time.now()
        cloud.header.frame_id = "map"
        cloud.points.append(point)
        self.target_local_debug.publish(cloud)

        # Перевод в глобальные координаты в пространстве
        telemetry = self.get_telemetry()
        x_norm = np.array([1, 0, 0])
        y_norm = np.array([0, 1, 0])
        z_norm = np.array([0, 0, 1])

        turn_1 = turn_vector(x_norm, -z_norm, telemetry.yaw)
        axis = turn_vector(y_norm, -z_norm, telemetry.yaw)
        turn_2 = turn_vector(turn_1, -axis, telemetry.pitch)
        z_global_vector = turn_2.copy()
        x_global_vector = turn_vector(-axis, -z_global_vector, telemetry.roll)
        y_global_vector = np.cross(x_global_vector, z_global_vector)

        # Дебаг вывод локальной системы координат дрона
        point_x, point_y, point_z = Point32(), Point32(), Point32()
        x_global_vector_norm = x_global_vector / np.linalg.norm(x_global_vector) + self.get_position()
        y_global_vector_norm = y_global_vector / np.linalg.norm(y_global_vector) + self.get_position()
        z_global_vector_norm = z_global_vector / np.linalg.norm(z_global_vector) + self.get_position()
        point_x.x, point_x.y, point_x.z = x_global_vector_norm[0], x_global_vector_norm[1], x_global_vector_norm[2]
        point_y.x, point_y.y, point_y.z = y_global_vector_norm[0], y_global_vector_norm[1], y_global_vector_norm[2]
        point_z.x, point_z.y, point_z.z = z_global_vector_norm[0], z_global_vector_norm[1], z_global_vector_norm[2]
        cloud = PointCloud()
        cloud.header.stamp = rospy.Time.now()
        cloud.header.frame_id = "aruco_map"
        cloud.points.append(point_x)
        cloud.points.append(point_y)
        cloud.points.append(point_z)
        self.detection_axis_debug.publish(cloud)

        target_position = x_global_vector * x_local + y_global_vector * y_local + z_global_vector * z_local
        target_position += self.get_position()

        self.target_position.publish(f"{target_position[0]} {target_position[1]} {target_position[2]} {x_pix} {y_pix}")
        self.target_detections = self.target_detections[:min(len(self.target_detections), self.DEPTH_QUEUE_SIZE)]
        self.target_detections.append(
            {'timestamp': {'secs': secs, 'nsecs': nsecs}, 'position': target_position})
        self.consecutive_detections += 1

        # Дебаг вывод координат цели в пространстве
        point = Point32(target_position[0], target_position[1], target_position[2])
        cloud = PointCloud()
        cloud.header.stamp = rospy.Time.now()
        cloud.header.frame_id = "aruco_map"
        cloud.points.append(point)
        self.target_global_debug.publish(cloud)

    def run(self):
        rospy.spin()


class Telemetry:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

if __name__=="__main__":
    controller = TargetController()
    controller.run()