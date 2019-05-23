#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

ROOT_CAMERA_PATH = "/home/ubuntu/catkin_ws/src/lane_detection/src/lane_detection/camera_cal/"

def _get_camera_calibration():
    '''
    reads the calibrated values 
    
    Returns:
        It returns the camera matrix, distortion coefficients
    '''

    global ROOT_CAMERA_PATH

    dist_pickle = pickle.load(open( ROOT_CAMERA_PATH + "camera_cal.p", "rb" ))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx, dist


def _do_calibration():
    '''
    this is the function for doijng calibration and storing the result in the pickle file 
    for camera calibration and store the result in a file "camera_cal/camera_cal.p" 
    Array to store the obj point and image points
    
    Return:
        This function write a file to the disk which obtain the camera matrix, distortion coefficients .

    
    '''
    global ROOT_CAMERA_PATH

    objpoints = []
    imgpoints = []
    objp = np.zeros((6*9,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    images = glob.glob(ROOT_CAMERA_PATH + "/calibration*")
    for fnames in images:
        img = mpimg.imread(fnames)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            # plt.imshow(img)
            # plt.show()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    #save the calibration data in a pickle file to use later
    camera_cal_val = ROOT_CAMERA_PATH + "camera_cal.p" 
    output = open(camera_cal_val, 'wb')

    mydict2 = {'mtx': 1, 'dist': 2}
    mydict2['mtx'] = mtx
    mydict2['dist'] = dist
    pickle.dump(mydict2, output, protocol=2)
    output.close()


def undistort_image(distort_img, Visualization=False, Do_calibration=False):
    '''
    This function for undistoring images , obtain the undistorted images
    
    Args:
        distort_img, Visualization=False, if Visualization==True, will show the (un)disorted image
    

    Returns:
        The undistorted image

    '''

    if Do_calibration is True:
        _do_calibration()

    mtx, dist = _get_camera_calibration()
    # print "mtx: \n{0}\n dist:\n{1}\n".format( mtx, dist)
    if type(distort_img) == str:
        distort_img = cv2.imread(distort_img)
    
    distort_img = cv2.cvtColor(distort_img, cv2.COLOR_BGR2RGB)
    undist_image = cv2.undistort(distort_img, mtx, dist, None, mtx)
    
    if Visualization:
        plt.subplot(1,2,1)
        plt.imshow(distort_img)
        plt.subplot(1,2,2)
        plt.imshow(undist_image)
        plt.show()
        return undist_image
    else:
        # pass
        return undist_image
	


if __name__ == '__main__':
    undistort_image("/home/ubuntu/catkin_ws/src/lane_detection/test/test3.jpg", Visualization=True, Do_calibration=True)
