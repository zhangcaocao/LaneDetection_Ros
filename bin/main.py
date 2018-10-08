#!/usr/bin/python
# -*- coding: utf-8 -*-

import glob
import pickle
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
from std_msgs.msg import Float64
import time

QUEUE_SIZE = 3

class pipeline():
    def __init__(self):
        
        self.Result_Image_pub   = rospy.Publisher('Result_Image', sensor_msgs.msg.Image, queue_size=QUEUE_SIZE)
        self.Deviation_pub      = rospy.Publisher('Deviation', Float64, queue_size=QUEUE_SIZE)
        self.Avg_cur_pub        = rospy.Publisher('Avg_curvature', Float64, queue_size=QUEUE_SIZE)

    
    def _draw_on_original(self, undist, left_fitx, right_fitx, ploty,Minv):
        # Create an image to draw the lines on
        color_warp = np.zeros_like(undist).astype(np.uint8)
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane with low confidence region in red
        cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0))
        
        #confidence region in green
        # shift = 50
        diff = (right_fitx - left_fitx)/2
        pts_left = np.array([np.transpose(np.vstack([left_fitx[400:], ploty[400:]]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx[400:], ploty[400:]])))])
        pts = np.hstack((pts_left, pts_right))
        
        # cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        print np.int_([pts]).shape    
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        right_x_num = int(right_fitx.size/2)
        left_x_num = int(left_fitx.size/2)
        cv2.circle(color_warp, (int(left_fitx[left_x_num]),int(ploty[left_x_num])), 20, (0, 0, 255), -5)
        cv2.circle(color_warp, (int(right_fitx[right_x_num]),int(ploty[right_x_num])), 20, (0, 0, 255), -5)
        cv2.line(color_warp, (int(left_fitx[left_x_num]),int(ploty[left_x_num])), (int(right_fitx[right_x_num]), int(ploty[right_x_num])), (0, 255, 0), 5)
        cv2.line(color_warp, (int(color_warp.shape[1]/2),0), (int(color_warp.shape[1]/2), int(color_warp.shape[0])), (0, 255, 0), 2)
        
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
        # Combine the result with the original image

        result = cv2.addWeighted(undist, 1, newwarp, 0.4, 0)
        
        return result

    def _pipeline(self, img, model='debug'):
        # print "self._mag_thresh : {0}".format(type(list(self._mag_thresh)))
        undist = calibration_main.undistort_image(img, Visualization=False)
        # thresh_combined, grad_th, col_th 
        time_bin_1 = time.clock()
        final_combined, abs_bin, mag_bin, dir_bin, hls_bin  = thresholding_main.Threshold().combined_thresh(undist)
        perspective,unwarped ,m ,Minv = perspective_regionofint_main.perspective_transform(final_combined)
        #pass the perspective image to the lane fitting stage
        time_bin_2 = time.clock()
        slid_time_1 = time.clock()
        slides_pers, left_fitx, right_fitx, ploty, avg_cur, dist_centre_val = sliding_main.for_sliding_window(perspective)
        slid_time_2 = time.clock()
                
        #draw the detected lanes on the original image for_sliding_window
        _draw_on_original_time1 = time.clock()
        mapped_lane = self._draw_on_original(undist, left_fitx, right_fitx, ploty, Minv)
        _draw_on_original_time2 = time.clock()

        #font and text for drawing the offset and curvature 
        viz_time1 = time.clock()
        curvature = "Estimated lane curvature %.2fm" % (avg_cur)
        dist_centre = "Estimated offset from lane center %.6f  cm" % (dist_centre_val * 100)
        font = cv2.FONT_HERSHEY_COMPLEX
        
        if model == 'debug':
            middlepanel = np.zeros((120, 900, 3), dtype=np.uint8)
            l1 = np.zeros((50, 50, 3), dtype=np.uint8)
            l2 = np.zeros((50, 50, 3), dtype=np.uint8)
            l3 = np.zeros((50, 50, 3), dtype=np.uint8)
            l4 = np.zeros((50, 50, 3), dtype=np.uint8)
            l5 = np.zeros((50, 50, 3), dtype=np.uint8)
            l6 = np.zeros((50, 50, 3), dtype=np.uint8)
            l7 = np.zeros((50, 50, 3), dtype=np.uint8)
            legend = np.zeros((240, 1200, 3), dtype=np.uint8)

            cv2.putText(middlepanel, curvature, (30, 60), font, 1, (255,255,255), 2)
            cv2.putText(middlepanel, dist_centre, (30, 90), font, 1, (255,255,255), 2)
            cv2.putText(l1,"1", (15, 35), font, 1, (255,255,0), 2)
            cv2.putText(l2,"2", (15, 30), font, 1, (255,255,0), 2)
            cv2.putText(l3,"3", (15, 30), font, 1, (255,255,0), 2)
            cv2.putText(l4,"4", (15, 30), font, 1, (255,255,0), 2)
            cv2.putText(l5,"5", (15, 30), font, 1, (255,255,0), 2)
            cv2.putText(l6,"6", (15, 30), font, 1, (255,255,0), 2)
            cv2.putText(l7,"7", (15, 30), font, 1, (255,255,0), 2)
            text = "1-Detected Lanes, \
                2-Color Threshold\n3-Gradient Threshold,         4-Thresholding operations combined\n\
    5-Perspective Transformation,6-Original Frame\n\
    7-Mapping Polynomials,\n>Blue line-current frame polynomial fit, \n>Green line-smoothened polynomial fit, \
                    \n>Pink - Lane pixels"

            y0, dy = 30, 30
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                cv2.putText(legend, line, (20, y ), font, 1, (255,255,255),2)

            diagScreen                     = np.zeros((1080, 1920, 3), dtype=np.uint8)
            #2
            diagScreen[0:360, 1200:1560]   = cv2.resize(np.dstack((hls_bin*255,hls_bin*255, hls_bin*255)), (360,360), interpolation=cv2.INTER_AREA) 
            #3
            diagScreen[0:360, 1560:1920]   = cv2.resize(np.dstack((abs_bin*255,abs_bin*255,abs_bin*255)), (360,360), interpolation=cv2.INTER_AREA) 
            #4
            diagScreen[360:720, 1200:1560] = cv2.resize(np.dstack((final_combined*255,final_combined*255,final_combined*255)), (360,360), interpolation=cv2.INTER_AREA)  
            #5
            diagScreen[360:720,1560:1920]  = cv2.resize(np.dstack((perspective*255, perspective*255, perspective*255)), (360,360), interpolation=cv2.INTER_AREA) 
            #6
            diagScreen[720:1080,1200:1560] = cv2.resize(img, (360,360), interpolation=cv2.INTER_AREA) 
            #7
            diagScreen[720:1080,1560:1920] = cv2.resize(slides_pers, (360,360), interpolation=cv2.INTER_AREA) 
            #1
            diagScreen[0:720, 0:1200]      = cv2.resize(mapped_lane, (1200,720), interpolation=cv2.INTER_AREA) 

            diagScreen[720:840, 0:900]     = middlepanel
            diagScreen[0:50, 0:50]         = l1
            diagScreen[0:50, 1200: 1250]   = l2
            diagScreen[0:50, 1560:1610]    = l3
            diagScreen[720:770, 1560:1610] = l7
            diagScreen[360:410, 1560:1610] = l5
            diagScreen[720:770, 1200:1250] = l6
            diagScreen[360:410, 1200:1250] = l4
            diagScreen[840:1080, 0:1200]   = legend
            
            viz_time2 = time.clock()
            rospy.loginfo(
                "*****---  Used time: ***----\n"
                "bin_time:" + str(time_bin_2 - time_bin_1) +"   " 
             + "slid_time:" + str(slid_time_2 - slid_time_1) + "    " 
             + "_draw_on_original_time: " + str(_draw_on_original_time2 - _draw_on_original_time1) + "   "
             + "viz time:" + str(viz_time2 - viz_time1)
             )   
            return diagScreen, dist_centre_val, avg_cur

        cv2.putText(mapped_lane, curvature, (30, 60), font, 1, (255,0,0), 2)
        cv2.putText(mapped_lane, dist_centre, (30, 120), font, 1, (255,0,0), 2)
        viz_time2 = time.clock()
        rospy.loginfo(
                "*****---  Used time: ***----\n"
                "bin_time:" + str(time_bin_2 - time_bin_1) +"   " 
             + "slid_time:" + str(slid_time_2 - slid_time_1) + "    " 
             + "_draw_on_original_time: " + str(_draw_on_original_time2 - _draw_on_original_time1) + "   "
             + "viz time:" + str(viz_time2 - viz_time1)
             ) 
        return mapped_lane, dist_centre_val, avg_cur


    def _Test(self):
        image = cv2.imread('/home/zhangcaocao/catkin_ws/src/lane_detection/test/test.jpg')
        mapped_lane = _pipeline(image)
        plt.imshow(mapped_lane)
        plt.show()
    
    #回调函数输入的应该是msg
    def _callback(self,Image):
        try:
            cvb = CvBridge()
            cvimg = cvb.imgmsg_to_cv2(Image)
            time1 = time.clock()
            result, dist_centre_val, avg_cur = self._pipeline(cvimg)
            rospy.loginfo("dist_centre_val: " + str(dist_centre_val))
            rospy.loginfo("_pipeline:" + str(time.clock() - time1))        
            self.Result_Image_pub.publish(cvb.cv2_to_imgmsg(result))
            self.Avg_cur_pub.publish(avg_cur)
            # r < 0; l > 0 。
            self.Deviation_pub.publish(dist_centre_val)
            
        except CvBridgeError as e:
            rospy.logerr(e)


    def _listener(self):
        rospy.init_node('lane_detection_node', anonymous=True)
        #Subscriber函数第一个参数是topic的名称，第二个参数是接受的数据类型 第三个参数是回调函数的名称
        rospy.Subscriber('Image', sensor_msgs.msg.Image, self._callback, queue_size=QUEUE_SIZE)
        rospy.spin()

    def main(self):
        self._listener()

if __name__ == '__main__':
    lanedetection_pipeline = pipeline()
    lanedetection_pipeline.main()
