# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from global_lane import *
import rospy

class sliding_window():
    def __init__(self, Margin, Min_Size, nwindows, _binary_img):
        self.Lane_Detected = False
        self.Margin = Margin
        self.Min_Size = Min_Size
        self.Dist_Center_Val_List = []
        self.Pre_Dist_Center_Val = 0
        self.MinPix = Min_Size / nwindows
        self.nwindows = nwindows
        self.binary_img = _binary_img
        self.vis_img = np.dstack((_binary_img, _binary_img, _binary_img))*255
        self.window_height = np.int(_binary_img.shape[0]/ (2 * nwindows))

        # 这个里面记录了所有二值化图像的白色像素点的坐标;
        self.nozero_array = _binary_img.nonzero()
        self.X_nozero = np.array(self.nozero_array[0])
        self.Y_nozero = np.array(self.nozero_array[1])

        # 拟合曲线：
        self.left_polyfit = 0
        self.right_polyfit = 0
        self.ploty = np.linspace(0, self.vis_img.shape[0] - 1, self.vis_img.shape[0])
        self.left_fitx = 0
        self.right_fitx = 0
        
        self.left_lane = Line()
        self.right_lane = Line()
        self.left_lane_inds_list = []
        self.right_lane_inds_list = []
    
    def _averageList(self, list_):
        nsum = 0
        for i in range(len(list_)):
            nsum += list_[i]
        return nsum / len(list_)
    
    def _histogram_Process(self):
        histogram = np.sum(self.binary_img[int(self.binary_img.shape[0]/2):, :], axis=0)
        mid_point_val = np.int(histogram.shape[0]/2)
        leftx_histogram_val = np.argmax(histogram[:mid_point_val])
        rightx_histogram_val = np.argmax(histogram[mid_point_val:]) + mid_point_val

        return leftx_histogram_val, rightx_histogram_val
    
    def _slide_windows(self, leftx_histogram_val, rightx_histogram_val):
        leftx_cur_val = leftx_histogram_val
        rightx_cur_val = rightx_histogram_val



        for window in range(self.nwindows):
            Y_low_window_val        = self.binary_img.shape[0] - (window+1)*self.window_height
            Y_high_window_val       = self.binary_img.shape[0] - window*self.window_height
            Xleft_low_window_val    = leftx_cur_val  - self.Margin
            Xleft_high_window_val   = rightx_cur_val + self.Margin
            X_right_low_window_val  = rightx_cur_val - self.Margin
            X_right_high_window_val = rightx_cur_val + self.Margin

            cv2.rectangle(self.vis_img,(Xleft_low_window_val, Y_low_window_val),(Xleft_high_window_val, Y_high_window_val),(0,255,0), 2) 
            cv2.rectangle(self.vis_img,(X_right_low_window_val, Y_low_window_val),(X_right_high_window_val, Y_high_window_val),(0,255,0), 2) 

            # 把框框内的0去掉;
            good_left_inds  = ((self.Y_nozero > Y_low_window_val) & (self.Y_nozero < Y_high_window_val) & (self.X_nozero > Xleft_low_window_val) & (self.X_nozero < Xleft_high_window_val)).nonzero()[0]
            good_right_inds = ((self.Y_nozero > Y_low_window_val) & (self.Y_nozero < Y_high_window_val) & (self.X_nozero > X_right_low_window_val) & (self.X_nozero < X_right_high_window_val)).nonzero()[0]
            
            self.left_lane_inds_list.append(good_left_inds)
            self.right_lane_inds_list.append(good_right_inds)

            # 检验每个框框类的像素数量, 符合数量上的要求的话，，就取平均值。
            if len(good_left_inds)  > self.MinPix:
                leftx_cur_val = np.int(np.mean(self.X_nozero[good_left_inds]))
            if len(good_right_inds) > self.MinPix:        
                rightx_cur_val = np.int(np.mean(self.X_nozero[good_right_inds]))
        
        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[self.left_lane_inds]
        lefty = nonzeroy[self.left_lane_inds] 
        rightx = nonzerox[self.right_lane_inds]
        righty = nonzeroy[self.right_lane_inds]

        return leftx, lefty, rightx, righty

    def _ployfit_lane(self, leftx, lefty, rightx, righty):

        self.left_lane.detected = False
        self.right_lane.detected = False

        if leftx.size > self.Min_Size:
            self.left_polyfit = np.polyfit(lefty, leftx, 2)
            self.left_fitx  = left_polyfit[0]*self.ploty**2 + left_polyfit[1]*self.ploty + left_polyfit[2]
            self.left_lane.detected = True
            self.Lane_Detected = True
        elif rightx.size > self.Min_Size:
            self.right_polyfit = np.polyfit(righty, rightx, 2)
            self.right_fitx = right_polyfit[0]*self.ploty**2 + right_polyfit[1]*self.ploty + right_polyfit[2]
            self.right_lane.detected = True
            self.Lane_Detected = True
        else:
            self.Lane_Detected = False
            logerr("&&&&&  ------  Oh here haven`t lane!!")

        window_img = np.zeros_like(self.vis_img)
        self.left_lane.bestx = np.vstack([self.left_lane.bestx,self.left_fitx])
        self.right_lane.bestx = np.vstack([self.right_lane.bestx,self.right_fitx])
    
        left_line_window1 = np.array([np.transpose(np.vstack([left_lane.bestx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_lane.bestx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_lane.bestx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_lane.bestx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,0, 0))

        self.vis_img = cv2.addWeighted(self.vis_img, 1, window_img, 1, 0)

        return self.vis_img

    def _cal_dist_centre_val(self, vis_img):
        ym_per_pix = 0.005                #70/115       # meters per pixel in y dimension
        xm_per_pix = 0.0015            #0.13/83   # meters per pixel in x dimension 

        right_x_num = int(self.right_fitx.size/2)
        left_x_num = int(self.left_fitx.size/2)
        cv2.circle(vis_img, (int(self.left_fitx[left_x_num]),int(self.ploty[left_x_num])), 10, (0, 255, 0), -1)
        cv2.circle(vis_img, (int(self.right_fitx[right_x_num]),int(self.ploty[right_x_num])), 10, (0, 255, 0), -1)
        cv2.line(vis_img, (int(self.left_fitx[left_x_num]),int(self.ploty[left_x_num])), (int(right_fitx[right_x_num]), int(self.ploty[right_x_num])), (0, 255, 0), 5)
        cv2.line(vis_img, (int(vis_img.shape[1]/2),0), (int(vis_img.shape[1]/2), int(vis_img.shape[0])), (0, 0, 250), 2)
        camera_center = vis_img.shape[1]/2.0
        for i in range(25):    
            num = np.random.randint((right_fitx.size/2 - 25), (right_fitx.size/2 + 25))
            lane_center = (left_fitx[num] + right_fitx[num])/2.0    
            dist_centre_val = (lane_center - camera_center)*xm_per_pix
            self.Dist_Center_Val_List.append(dist_centre_val)
        dist_centre_val = averagenum(self.Dist_Center_Val_List)
        self.Pre_Dist_Center_Val = dist_centre_val

        return dist_centre_val, self.Pre_Dist_Center_Val, vis_img

    def sliding_windows(self):

        leftx_histogram_val, rightx_histogram_val = self._histogram_Process()
        leftx, lefty, rightx, righty = self._slide_windows(leftx_histogram_val, rightx_histogram_val)
        out_img = _ployfit_lane(leftx, lefty, rightx, righty)
        dist_centre_val, Pre_Dist_Center_Val, result_img = _cal_dist_centre_val(out_img)

        return self.ploty, dist_centre_val, slef.Lane_Detected, result_img





















