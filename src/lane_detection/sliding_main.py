# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from global_lane import *
import rospy

LANE_DETECTED = False

MARGIN = 10
MIN_SIZE = 70
LANE_WIDTH_list = [] 
dist_centre_val_list = []
pre_dist_centre_val = 0

left_fitx = 0
right_fitx = 0

def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

def for_sliding_window(binary_warped):
    global LANE_DETECTED
    global MARGIN
    global LANE_WIDTH_list 
    global dist_centre_val_list
    global pre_dist_centre_val

    global right_fitx
    global left_fitx

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the imagelll
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and roohjdfskklj


    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    '''
        lane_width_detected = abs(leftx_base - rightx_base)
        print "leftx_base - rightx_base",  lane_width_detected
        if len(LANE_WIDTH_list) > 10:
            LANE_WIDTH_list.pop(0)
        LANE_WIDTH_list.append(lane_width_detected)
        lane_width_ave = averagenum(LANE_WIDTH_list)
    '''

    # Choose the number of sliding windows
    nwindows = 6
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/ (2 * nwindows))
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = MARGIN
    # Set minimum number of pixels found to recenter window
    minpix = (MIN_SIZE / nwindows)
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 


    #check for any lanes that are not detected in this current frame then use the history
    if (leftx.size < MIN_SIZE):
        left_lane.detected = False
    else:
        left_lane.detected = True
    
    if (rightx.size < MIN_SIZE):
        right_lane.detected = False
    else:
        right_lane.detected = True

    if left_lane.detected == True or right_lane.detected == True:
        LANE_DETECTED = True
    else:
        LANE_DETECTED = False 

    '''
    拟合多项式, 获取多项式上面的像素点;

    @_1：两条线都检测到，都进行拟合;
    @_2：只有一条检测到：只拟合一条, 返回一个标志。
    '''
    #if all lane is detected then try to fit the poly
    if left_lane.detected == True and right_lane.detected == True:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # 拟合的多项式：
        ploty      = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
        left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        # print type(left_fitx)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        left_lane.best_fit = np.vstack([left_lane.best_fit,left_fit])
        left_lane.best_fit[0] = left_fit
        right_lane.best_fit = np.vstack([right_lane.best_fit,right_fit])
        right_lane.best_fit[0] = right_fit
        left_lane.best_fit = np.average(left_lane.best_fit[-left_lane.smoothen_nframes:], axis = 0)
        right_lane.best_fit = np.average(right_lane.best_fit[-right_lane.smoothen_nframes:], axis = 0)
    else: 
        if left_lane.detected == True and right_lane.detected == False:
            left_fit = np.polyfit(lefty, leftx, 2)
            ploty      = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
            left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            left_lane.best_fit = np.vstack([left_lane.best_fit,left_fit])
            left_lane.best_fit[0] = left_fit
            left_lane.best_fit = np.average(left_lane.best_fit[-left_lane.smoothen_nframes:], axis = 0)

        if left_lane.detected == False and right_lane.detected == True:
            right_fit  = np.polyfit(righty, rightx, 2)
            ploty      = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            right_lane.best_fit     = np.vstack([right_lane.best_fit,right_fit])
            right_lane.best_fit[0]  = right_fit
            right_lane.best_fit     = np.average(right_lane.best_fit[-right_lane.smoothen_nframes:], axis = 0)
        else:
            '''
            一条线都没有检测到
            '''

            rospy.logwarn("&&&&& ------ Lane Not Detected!!")
            #use the history avg values 
            left_fit = left_lane.best_fit
            right_fit = right_lane.best_fit
            #print ("used previous best fit")


    if left_lane.first_frame == True:
        left_lane.first_frame = False
        left_lane.bestx = np.vstack([left_lane.bestx,left_fitx])
        left_lane.bestx[0] = left_fitx
    if right_lane.first_frame == True:
        right_lane.first_frame = False
        right_lane.bestx = np.vstack([right_lane.bestx,right_fitx])
        right_lane.bestx[0] = right_fitx
    try:
        left_lane.bestx = np.vstack([left_lane.bestx,left_fitx])
        left_lane.bestx = np.average(left_lane.bestx[-left_lane.smoothen_nframes:], axis = 0)
        right_lane.bestx = np.vstack([right_lane.bestx,right_fitx])
        right_lane.bestx = np.average(right_lane.bestx[-right_lane.smoothen_nframes:], axis = 0)
    except Exception as identifier:
        rospy.logerr(identifier)
        

    
    '''
    line 184 to line 204  单纯的可视化
    '''
    window_img = np.zeros_like(out_img)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 255]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_lane.bestx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_lane.bestx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_lane.bestx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_lane.bestx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,0, 0))

    result = cv2.addWeighted(out_img, 1, window_img, 1, 0)
    camera_centre = result.shape[1]/2.0

    print "leftx.size {0},rightx.size {1}".format(leftx.size, rightx.size)
    #if the lane was detectded then calculate the curvatire or use the history
    if (leftx.size > MIN_SIZE and rightx.size > MIN_SIZE ):

        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 0.005                #70/115       # meters per pixel in y dimension
        xm_per_pix = 0.0015            #0.13/83   # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        right_x_num = int(right_fitx.size/2)
        left_x_num = int(left_fitx.size/2)
        cv2.circle(result, (int(left_fitx[left_x_num]),int(ploty[left_x_num])), 5, (0, 255, 0), -1)
        cv2.circle(result, (int(right_fitx[right_x_num]),int(ploty[right_x_num])), 5, (0, 255, 0), -1)
        cv2.line(result, (int(left_fitx[left_x_num]),int(ploty[left_x_num])), (int(right_fitx[right_x_num]), int(ploty[right_x_num])), (0, 255, 0), 5)
        cv2.line(result, (int(result.shape[1]/2),0), (int(result.shape[1]/2), int(result.shape[0])), (0, 0, 250), 2)
        

        print 'abs(left_fitx[num] - right_fitx[num])', abs(left_fitx[right_x_num] - right_fitx[left_x_num])
        
        if (abs(left_fitx[right_x_num] - right_fitx[left_x_num]) <= 3):
            dist_centre_val = 30000 * pre_dist_centre_val
            rospy.logerr(" &&&&&&&&&&&&&&& Noly One lane detected!! ")
        else:
            # 取50个点计算误差，平均, 并且记录之前的误差值
            for i in range(50):    
                num = np.random.randint((right_fitx.size/2 - 25), (right_fitx.size/2 + 25))
                lane_centre = (left_fitx[num] + right_fitx[num])/2.0    
                dist_centre_val = (lane_centre - camera_centre)*xm_per_pix
                dist_centre_val_list.append(dist_centre_val)
            dist_centre_val = averagenum(dist_centre_val_list)
            pre_dist_centre_val = dist_centre_val
        # 右边 < 0; 左边 > 0 。
        avg_cur = (right_curverad+left_curverad)/2.0

        
        left_lane.line_base_pos = np.vstack([left_lane.line_base_pos,dist_centre_val])
        left_lane.line_base_pos[0] = dist_centre_val
        left_lane.line_base_pos = np.average(left_lane.line_base_pos[-left_lane.smoothen_nframes:], axis = 0)
        
        left_lane.radius_of_curvature = np.vstack([left_lane.radius_of_curvature,avg_cur])
        left_lane.radius_of_curvature[0] = avg_cur
        left_lane.radius_of_curvature = np.average(left_lane.radius_of_curvature[-left_lane.smoothen_nframes:], axis = 0)
    # else use the history curvature
    else:
        if leftx.size > MIN_SIZE or rightx.size > MIN_SIZE:
            rospy.logerr(" ----------- Noly One lane detected!! ")
            # dist_centre_val = 30000 * pre_dist_centre_val
            num = np.random.randint((right_fitx.size/2 - 25), (right_fitx.size/2 + 25))
            if leftx.size > MIN_SIZE:
                dist_centre_val  = 30000
            elif rightx.size > MIN_SIZE:
                dist_centre_val = -30000
        avg_cur = left_lane.radius_of_curvature
    
    #reset the lane detected to false for the next frame 
    left_lane.detected == False
    right_lane.detected == False
    return result, left_lane.bestx, right_lane.bestx, ploty, left_lane.radius_of_curvature, dist_centre_val, LANE_DETECTED
