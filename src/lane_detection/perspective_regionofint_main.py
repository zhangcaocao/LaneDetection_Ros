#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

import calibration_main
import thresholding_main


def perspective_transform(img):
    '''
    perspective transform
    '''
    
    # img = cv2.resize(img ,(240, 320), interpolation=cv2.INTER_CUBIC)
    # plt.imshow(img)
    # plt.show()

    imshape = img.shape
    print (imshape)

    vertices = np.array([[(imshape[1], 0.63*imshape[0]), (imshape[1], imshape[0]),
                       (0.42*imshape[1],imshape[0]),(0.42*imshape[1], 0.63*imshape[0])]], dtype=np.float32)
    src= np.float32(vertices)
    dst = np.float32([[img.shape[1],0],[img.shape[1],img.shape[0]],
                      [0.25*img.shape[1],img.shape[0]],[0.25*img.shape[1],0]])

    
    m = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (imshape[1], imshape[0]) 
    perspective_img = cv2.warpPerspective(img, m, img_size, flags = cv2.INTER_LINEAR)  
    unwarped = cv2.warpPerspective(perspective_img, minv, (perspective_img.shape[1], perspective_img.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG  
    return perspective_img, unwarped,m ,minv


def region_of_interest(img, vertices):
    '''
    #region of interest
    '''
    #defining a blank mask to start with
    mask = np.zeros_like(img, dtype=np.uint8)
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


if __name__ == '__main__':

    img_file = '/home/zhangcaocao/catkin_ws/src/lane_detection/test/test1.jpg'
    img = calibration_main.undistort_image(img_file, Visualization=False)
    # img, abs_bin, mag_bin, dir_bin, hls_bin = thresholding_main.combined_thresh(img)
    
    warped, unwarped, m, m_inv = perspective_transform(img)
    
    plt.subplot(1, 2, 1)
    # unwarped = cv2.resize(unwarped ,(240, 320), interpolation=cv2.INTER_CUBIC)
    plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)

    plt.subplot(1, 2, 2)
    # warped = cv2.resize(warped ,(240, 320), interpolation=cv2.INTER_CUBIC)
    plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
    plt.show()

    '''
    Error :
        The image's shape will be changed if using the "plt.savefig('warped.png')"
    plt.imshow(warped, cmap='gray', vmin=0, vmax=1, shape=(600, 800))
    plt.axis('off') 
    plt.savefig('warped.png')
    print warped.shape
    plt.show()
    '''

    cv2.imwrite('warped.png', warped)
    
    print warped.shape
    # plt.show()
