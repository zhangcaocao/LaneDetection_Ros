#!/usr/bin/python
import math
import sys
import time

import cv2
import numpy
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError, getCvType



QUEUE_SIZE = 10

# Send each image by iterate it from given array of files names  to a given topic,
# as a regular and compressed ROS Images msgs.
class Source:

    def __init__(self, topic, cam):
        self.pub = rospy.Publisher(topic, sensor_msgs.msg.Image, queue_size=QUEUE_SIZE)
        self.cap = cv2.VideoCapture(int(cam))

    def spin(self):
        cvb = CvBridge()
        while not rospy.core.is_shutdown():
            ret, cvim = self.cap.read()
            # print(ret)
            # cvim = cv2.imread('/home/zhangcaocao/catkin_ws/src/lane_detection/test/test2.jpg')
            cvim = cv2.resize(cvim ,(60, 80), interpolation=cv2.INTER_CUBIC)
            rate = rospy.Rate(30)
            rospy.loginfo("image shape: " + str(cvim.shape))
            self.pub.publish(cvb.cv2_to_imgmsg(cvim))
            rate.sleep()

def main(args):
    s = Source('Image', args[1])
    rospy.init_node('Source')
    
    try:
        s.spin()
        rospy.spin()
        outcome = 'test completed'
    except KeyboardInterrupt:
        print "shutting down"
        outcome = 'keyboard interrupt'
    rospy.core.signal_shutdown(outcome)

if __name__ == '__main__':
    main(sys.argv)
