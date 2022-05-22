import RPi.GPIO as gpio
import rospy
from std_msgs.msg import Bool

node_name = "LoadCellController"

class LoadCellController:
    def __init__(self):
        rospy.init_node(node_name)
        rospy.loginfo(node_name + " started")
        self.pub = rospy.Publisher("load_cell/catch", Bool, queue_size=10)
        # Пины для получения информации с датчика и параметры для определения веса
        self.DT = 27
        self.SCK = 17
        self.DRONE_WEIGHT = 135
        self.WEIGHT_DELIMETER = 283
    # Считывание данных с датчика веса
    def readCount(self):
        i = 0
        Count = 0
        gpio.setup(self.DT, gpio.OUT)
        gpio.setup(self.SCK, gpio.OUT)
        gpio.output(self.DT, 1)
        gpio.output(self.SCK, 0)
        gpio.setup(self.DT, gpio.IN)

        while gpio.input(self.DT) == 1:
            i = 0
        for i in range(24):
            gpio.output(self.SCK, 1)
            Count = Count << 1

            gpio.output(self.SCK, 0)
            # time.sleep(0.001)
            if gpio.input(self.DT) == 0:
                Count = Count + 1

        gpio.output(self.SCK, 1)
        Count = Count ^ 0x800000
        gpio.output(self.SCK, 0)
        return Count
    # Основной цикл программы
    def run(self):
        gpio.setmode(gpio.BCM)
        sample = self.readCount()
        try:
            r = rospy.Rate(5)
            while not rospy.is_shutdown():
                count = self.readCount()
                weight = (sample - count) / self.WEIGHT_DELIMETER
                print(weight)
                # Если вес на датчике больше чем вес коптера
                if weight >= self.DRONE_WEIGHT:
                    self.pub.publish(True)
                r.sleep()
        except KeyboardInterrupt:
            rospy.signal_shutdown("Interrupted by keyboard")


if __name__ == '__main__':
    controller = LoadCellController()
    controller.run()




