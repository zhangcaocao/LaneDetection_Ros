#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

import calibration_main
import rospy
from lane_detection import *


class Threshold():
	def __init__(self):

		self._abs_sobel_thresh_val_min 	= (rospy.get_param("/lane_detection_node/_abs_sobel_thresh_min", None))
		self._abs_sobel_thresh_val_max 	= (rospy.get_param("/lane_detection_node/_abs_sobel_thresh_max", None))

		self._mag_thresh_val_min 		= (rospy.get_param("/lane_detection_node/_mag_thresh_min", None))
		self._mag_thresh_val_max 		= (rospy.get_param("/lane_detection_node/_mag_thresh_max", None))

		self._dir_thresh_val_min 		= (rospy.get_param("/lane_detection_node/_dir_threshold_min", None))
		self._dir_thresh_val_max 		= (rospy.get_param("/lane_detection_node/_dir_threshold_max", None))

		self._hls_thresh_val_min 		= (rospy.get_param("/lane_detection_node/_hls_thresh_min", None))
		self._hls_thresh_val_max 		= (rospy.get_param("/lane_detection_node/_hls_thresh_max", None))
		


	def _abs_sobel_thresh(self, img, orient='x', thresh_min=20, thresh_max=100):
		'''
		Takes an image, gradient orientation, and threshold min/max values

		Args:
			img: src image
			orient:Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
			thresh_min/thresh_max : gradient threshold 

		Return:
			abs_bin_image: The binary image 
		'''

		# Apply x or y gradient with the OpenCV Sobel() function
		# and take the absolute value
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		if orient == 'x':
			abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
		if orient == 'y':
			abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
		# Rescale back to 8 bit integer
		scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
		# Create a copy and apply the threshold
		abs_bin_image = np.zeros_like(scaled_sobel)
		# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
		abs_bin_image[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

		# Return the result
		return abs_bin_image

	def _mag_thresh(self, img, sobel_kernel=3, _mag_thresh=(30, 100)):
		'''
		Return the magnitude of the gradient for a given sobel kernel size and threshold values
		'''

		# Convert to grayscael
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		gradmag = np.sqrt(sobelx**2 + sobely**2)
		# Rescale to 8 bit
		scale_factor = np.max(gradmag)/255
		gradmag = (gradmag/scale_factor).astype(np.uint8)
		# Create a binary image of ones where threshold is met, zeros otherwise
		mag_bin_image = np.zeros_like(gradmag)
		mag_bin_image[(gradmag >= _mag_thresh[0]) & (gradmag <= _mag_thresh[1])] = 1
		return mag_bin_image


	def _dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
		'''
		Return the direction of the gradient for a given sobel kernel size and threshold values

		'''
		# print thresh[1]
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		# Take the absolute value of the gradient direction,
		# apply a threshold, and create a binary image result
		absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
		dir_bin_image =  np.zeros_like(absgraddir)
		dir_bin_image[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

		return dir_bin_image


	def _hls_thresh(self, img, thresh=(113, 255)):
		'''
		Convert RGB to HLS and threshold to binary image using S channel

		'''
		hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		s_channel = hls[:,:,2]
		hls_bin_image = np.zeros_like(s_channel)
		hls_bin_image[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
		return hls_bin_image


	def combined_thresh(self, img):
		'''
		This function Combine all the solutions and return the result separately
		
		'''
		imshape = img.shape
		# print type(img)

		
		abs_bin = self._abs_sobel_thresh(img, orient='x', thresh_min=self._abs_sobel_thresh_val_min, thresh_max=self._abs_sobel_thresh_val_max)
		
		mag_bin = self._mag_thresh(img, sobel_kernel=3, _mag_thresh=tuple((self._mag_thresh_val_min, self._mag_thresh_val_max)))
		dir_bin = self._dir_threshold(img, sobel_kernel=15, thresh=tuple((self._dir_thresh_val_min, self._dir_thresh_val_max)))
		hls_bin = self._hls_thresh(img, thresh=tuple((self._hls_thresh_val_min, self._hls_thresh_val_max)))

		final_combined = np.zeros_like(dir_bin)
		#final_combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1)))] = 1
                final_combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_bin == 1] = 1

		return final_combined, abs_bin, mag_bin, dir_bin, hls_bin  


if __name__ == '__main__':


	img_file = '/home/ubuntu/catkin_ws/src/lane_detection/test/test1.jpg'
	img = calibration_main.undistort_image(img_file, Visualization=False)
	all_combined, abs_bin, mag_bin, dir_bin, hls_bin = Threshold().combined_thresh(img)
	plt.subplot(3, 3, 1)
	plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 2)
	plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 3)
	plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 4)
	plt.imshow(hls_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 5)
	plt.imshow(img)
	plt.subplot(3, 3, 6)
	plt.imshow(all_combined, cmap='gray', vmin=0, vmax=1)
	plt.tight_layout()
	plt.show()
