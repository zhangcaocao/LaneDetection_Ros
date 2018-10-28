#!/usr/bin/python
# -*- coding: utf-8 -*-

QUEUE_SIZE = 3
DISTANCE = 10
LOAD_WIDTH = 6
MAX_SteeringAngle = 30


from geometry_msgs.msg import Twist, TwistStamped
from lane_detection import *
import rospy
from std_msgs.msg import Float64

import time


MAX_STEER = 1800

class PID_NODE():
    def __init__(self, KP, KI, KD):

        self.kp = KP
        self.ki = KI
        self.kd = KD
        self.p_error = 0.0
        self.d_error = 0.0
        self.i_error = 0.0
        self.prev_cte = 0.0
        self.sum_cte = 0.0
        self.SteeringAngle_CMD = TwistStamped()
        self.SteeringAngle_CMD.twist.angular.z = 0.0
        self.SteeringAngle_CMD.twist.linear.x  = 0.3
        self.vel_pub = rospy.Publisher("/cmd_vel", TwistStamped, queue_size = 1) 



    def _PID_Cal(self, cte):
        try:
            global MAX_STEER
            cte = cte.data
            self.p_error = cte
            self.d_error = cte - self.prev_cte
            self.i_error = self.sum_cte
            self.prev_cte = cte
            self.sum_cte += cte

            steer = -(self.kp * self.p_error + self.ki * self.i_error + self.kd * self.d_error)
            print "steer {0}".format(steer)
            if abs(steer) > MAX_STEER:
                if steer > 0.0:
                    steer = 1800.0
                if steer < 0.0:
                    steer = -1800.0 # (-1800, 1800)
            # (-0.5, 0.5) --- +0.5 -> (0, 1)
            normedsteeringAngle = 1- (steer / 1800 + 0.5)
            self.SteeringAngle_CMD.twist.angular.z = normedsteeringAngle
            self.vel_pub.publish(self.SteeringAngle_CMD)
            print "normedsteeringAngle {0}".format(normedsteeringAngle)
            print "------------"
        except Exception as e:
            print(e)

    def _callback(self, Deviation):
        try:
            self._PID_Cal(Deviation)

        except Exception as e:
            print(e)
                
    def _listener(self):
        rospy.init_node('PID_Twist_NODE', anonymous=True)
        #Subscriber函数第一个参数是topic的名称，第二个参数是接受的数据类型 第三个参数是回调函数的名称
        rospy.Subscriber('Deviation', Float64, self._callback, queue_size=QUEUE_SIZE)
        rospy.spin()

def main():
    pidnode = PID_NODE(60000, 0.01, 0.01)
    pidnode._listener()

    
if __name__ == '__main__':
    main()
    
