#!/usr/bin/python
# -*- coding: utf-8 -*-

import glob
import pickle
import time
from math import pi

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge, CvBridgeError, getCvType
from geometry_msgs.msg import Twist
from lane_detection import *
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

from pid_node import *

QUEUE_SIZE = 1

number = 0

class pipeline():
    def __init__(self):
        
        self.Result_Image_pub   = rospy.Publisher('Result_Image', sensor_msgs.msg.Image, queue_size=QUEUE_SIZE)
        self.Deviation_pub      = rospy.Publisher('Deviation', Float32, queue_size=QUEUE_SIZE)
        self.Avg_cur_pub        = rospy.Publisher('Avg_curvature', Float32, queue_size=QUEUE_SIZE)
        self.Lan_detected       = False
        self.pidnode            = PID_NODE(0.6, 0, 0.1)

    def _pipeline(self, img, model='debug'):

        # print "self._mag_thresh : {0}".format(type(list(self._mag_thresh)))
        # undist = calibration_main.undistort_image(img, Visualization=False)
        
        imshape = img.shape
        vertices = np.array([[(imshape[1], 0.6*imshape[0]), (imshape[1],0.9*imshape[0]),
                        (.0*imshape[1],0.9*imshape[0]),(.0*imshape[1], 0.6*imshape[0])]], dtype=np.int32)
        img , mask= perspective_regionofint_main.region_of_interest(img, vertices=vertices)
        undist = img
        # thresh_combined, grad_th, col_th 
        final_combined, abs_bin, mag_bin, dir_bin, hls_bin  = thresholding_main.Threshold().combined_thresh(undist)
        perspective,unwarped ,m ,Minv = perspective_regionofint_main.perspective_transform(final_combined)
        #pass the perspective image to the lane fitting stage
        # ploty, dist_centre_val, self.Lan_detected, mapped_lane = sliding_main.sliding_window(10, 150, 6, _binary_img = perspective).sliding_windows()
        slides_pers, left_fitx, right_fitx, ploty, avg_cur, dist_centre_val,  self.Lan_detected = sliding_main.for_sliding_window(perspective)
        
        #draw the detected lanes on the original image for_sliding_window

         
        # = self._draw_on_original(undist, left_fitx, right_fitx, ploty, Minv)
        mapped_lane = slides_pers.astype(np.uint8)
        # mapped_lane = undist
        #font and text for drawing the offset and curvature 
        viz_time1 = time.clock()
        curvature = "Estimated lane curvature %.2fm" % (avg_cur)
        dist_centre = "Estimated offset from lane center %.6f  cm" % (dist_centre_val * 100)
        font = cv2.FONT_HERSHEY_COMPLEX
        
        if model == 'debug':
            diagScreen                       = np.zeros((120,160, 3), dtype=np.uint8)  
            diagScreen[0:60, 80:160]         = cv2.resize(np.dstack((perspective*255, perspective*255, perspective*255)), (80,60), interpolation=cv2.INTER_AREA) 
            diagScreen[60:120, 0:80]         = cv2.resize(np.dstack((final_combined*255,final_combined*255,final_combined*255)), (80,60), interpolation=cv2.INTER_AREA)  
            diagScreen[0:60:,0:80]           = cv2.resize(img, (80,60), interpolation=cv2.INTER_AREA) 
            diagScreen[60:120, 80:160]       = cv2.resize(mapped_lane, (80,60), interpolation=cv2.INTER_AREA) 

            # return diagScreen, dist_centre_val, avg_cur

            # cv2.putText(diagScreen, 'perspective ', (0, 0), font, 1, (255,0,0), 1)
            # cv2.putText(diagScreen, 'final_combined', (60, 120), font, 1, (255,0,0), 1)

            # cv2.putText(diagScreen, 'img', (30, 60), font, 1, (255,0,0), 1)
            # cv2.putText(diagScreen, 'mapped_lane', (30, 120), font, 1, (255,0,0), 1)
            return diagScreen, dist_centre_val, avg_cur
        return mapped_lane, dist_centre_val, avg_cur



    def _Test(self):
        image = cv2.imread('/home/zhangcaocao/catkin_ws/src/lane_detection/test/test.jpg')
        mapped_lane = _pipeline(image)
        plt.imshow(mapped_lane)
        plt.show()
    
    
    #回调函数输入的应该是msg
    def _callback(self,Image):
        global number
        try:
            cvb = CvBridge()
            cvimg = cvb.imgmsg_to_cv2(Image)
            time1 = time.clock()
            result, dist_centre_val, avg_cur = self._pipeline(cvimg, 'debug')
            
            rospy.loginfo("dist_centre_val:  {0}   ".format(dist_centre_val * 1000))      
            self.Result_Image_pub.publish(cvb.cv2_to_imgmsg(result))
            if (self.lan_detected()):
                # number += 1
                self.Avg_cur_pub.publish(avg_cur)

                # r < 0; l > 0 。
                # print number
                # 减少小数点，提高计算速度; 
                self.Deviation_pub.publish(dist_centre_val * 1000)
                self.pidnode.PID_Cal(round((dist_centre_val * 1000), 3))
                
            else:
                rospy.logerr("---------- not detetced lane -------------")
            rospy.loginfo("Time :  " + str(time.clock() - time1))  
        except CvBridgeError as e:
            rospy.logerr(e)


    def _listener(self):
        rospy.init_node('lane_detection_node', anonymous=True)
        #Subscriber函数第一个参数是topic的名称，第二个参数是接受的数据类型 第三个参数是回调函数的名称
        rospy.Subscriber('/Image', sensor_msgs.msg.Image, self._callback, queue_size=QUEUE_SIZE)
        rospy.spin()

    def lan_detected(self):

        return self.Lan_detected

    def main(self):
        self._listener()


if __name__ == '__main__':
    lanedetection_pipeline = pipeline()
    lanedetection_pipeline.main()
