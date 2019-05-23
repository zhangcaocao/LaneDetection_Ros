#!/usr/bin/python
# -*- coding: utf-8 -*-

QUEUE_SIZE = 1
MAX_STEER = 16.0



from geometry_msgs.msg import Twist, TwistStamped
from lane_detection import *
import rospy
from std_msgs.msg import Float32

import time



class PID_NODE():
    def __init__(self, KP, KI, KD):

        self.kp = KP
        self.ki = KI
        self.kd = KD
        self.p_error = 0.0
        self.d_error = 0.0
        self.i_error = 0.0
        self.cte     = 0.0
        self.prev_cte = 0.0
        self.sum_cte  = 0.0
        self.number = 0
        self.SteeringAngle_CMD = Twist()
        self.SteeringAngle_CMD.angular.z = 0.0
        self.SteeringAngle_CMD.linear.x  = 0.15
        self.Lane_Steer_pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1) 
        


    def Update_Err(self, cte):
        try:
            global MAX_STEER
            self.cte = cte
            self.p_error = self.cte
            self.d_error = self.cte - self.prev_cte
            self.i_error = self.sum_cte
            self.prev_cte = self.cte
            self.sum_cte += self.cte
            rospy.loginfo('** Error Updated ! ' + "cte.data: {0}".format(self.cte))
            return self.p_error, self.i_error, self.d_error
        except Exception as e:
            rospy.logerr(e)
            # return self.p_error, self.i_error, self.d_error


    def PID_Cal(self, cte):
        try:
            global MAX_STEER
            p_error, i_error, d_error = self.Update_Err(cte)
            steer = -(self.kp * p_error + self.ki * i_error + self.kd * d_error)
            # print "steer {0}".format(steer)
            if abs(steer) > MAX_STEER:
                if steer > 0.0:
                    steer = MAX_STEER
                if steer < 0.0:
                    steer = -MAX_STEER

            # (-0.5, 0.5) --- +0.5 -> (0, 1)
            #    0   0.5  1  
            normedsteeringAngle = (steer / (5.1 * MAX_STEER))
            self.SteeringAngle_CMD.angular.z = normedsteeringAngle
            self.Lane_Steer_pub.publish(self.SteeringAngle_CMD)
            rospy.loginfo("normedsteeringAngle: {0}".format(normedsteeringAngle))


        except Exception as e:
            rospy.logerr(e)

#  rosrun rosserial_python serial_node.py /dev/ttyUSB0
