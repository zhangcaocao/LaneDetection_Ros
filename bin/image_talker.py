#!/usr/bin/python
import math
import sys
import time

import cv2
import numpy
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError, getCvType



QUEUE_SIZE = 1

# Send each image by iterate it from given array of files names  to a given topic,
# as a regular and compressed ROS Images msgs.
class Source:

    def __init__(self, topic, cam):
        self.pub            = rospy.Publisher(topic, sensor_msgs.msg.Image, queue_size=QUEUE_SIZE)
        self.pub_compressed = rospy.Publisher(topic + "/compressed", sensor_msgs.msg.CompressedImage, queue_size=QUEUE_SIZE)
        self.cap = cv2.VideoCapture(cam)

    def spin(self):
        cvb = CvBridge()
        while not rospy.core.is_shutdown():
            ret, cvim = self.cap.read()
            #image = cv2.imread('/home/zhangcaocao/catkin_ws/src/lane_detection/test/test.jpg')
            cvim = cv2.resize(cvim ,(240, 320), interpolation=cv2.INTER_CUBIC)
            rospy.loginfo("image shape: " + str(cvim.shape))
            self.pub.publish(cvb.cv2_to_imgmsg(cvim))
            self.pub_compressed.publish(cvb.cv2_to_compressed_imgmsg(cvim))
            time.sleep(0.1)

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
