#!/usr/bin/env python3
from controllers import *
import cv2
import numpy as np
from numpy import linalg as LA
from scipy.ndimage import filters

class Harris(object):
    
    def __init__(self, image, window_size, sigma, threshold):
        #h, w, _ = image.shape
        #image = cv2.resize(image, (h//2, w//2))
        self.image = image
        self.window_size = window_size
        self.sigma = sigma
        self.threshold = threshold
    
    def gradient_matrix(self):
        h, w, _ = self.image.shape
        self.gradient_matrix = np.zeros((h, w, 4))
        for y in range(h):
            for x in range(w):
                d_x = float(self.I_x[y, x])
                d_y = float(self.I_y[y, x])
                
                # for the position get length and the angle
                # between derivatives in the range od -pi to pi
                self.gradient_matrix[y, x] = [
                        d_x,
                        d_y,
                        LA.norm([d_x, d_y]),
                        cv2.fastAtan2(d_y, d_x)
                        ]
    
    def adaptive_non_max_suppression(mask):
        r = 0
        max_value = np.max(mask)
    
    
    def harris_matrix(self, f_name):
        
        def is_edge_close(h, w, y, x):
            is_edge_close = False
            if (y - 16) <= 0 or (y + 16) >= h:
                is_edge_close =  True
            if (x - 16) <= 0 or (x + 16) >= w:
                is_edge_close =  True
            
            return is_edge_close
        
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # DoG filtering
        kernel_dog = d_o_g(21, 1.5, 1.0)
        gray_image = cv2.filter2D(gray_image, -1, kernel_dog)
        cv2.imwrite('dog_gray.jpg', gray_image)
        
        self.I_x, self.I_y = compute_derivatives(gray_image)
        I_xx = self.I_x ** 2
        I_yy = self.I_y ** 2
        I_xy = self.I_x * self.I_y
        
        h, w, _ = self.image.shape
        kernel = GaussianKernel(self.window_size, self.sigma)
        #kernel = d_o_g(self.window_size, 1, 5)
        corner_list = []
        self.key_points = []
        corners_map = np.zeros((h,w))
        
        I_xx = np.array(cv2.filter2D(I_xx, -1, kernel), dtype = np.float32)
        I_yy = np.array(cv2.filter2D(I_yy, -1, kernel), dtype = np.float32)
        I_xy = np.array(cv2.filter2D(I_xy, -1, kernel), dtype = np.float32)
        
        # compute Harris feature strength, avoiding divide by zero
        imgH = (I_xx * I_yy - I_xy**2) / (I_xx + I_yy + 1e-8)
        
        # exclude points near the image border
        imgH[:16, :] = 0
        imgH[-16:, :] = 0
        imgH[:, :16] = 0
        imgH[:, -16:] = 0
            
        max_value = np.max(imgH)
        self.threshold *= max_value
        suppress_pos = imgH < self.threshold
        imgH[suppress_pos] = 0.0
        
        # non-maximum suppression in 5x5 regions
        maxH = filters.maximum_filter(imgH, (5,5))
        imgH = imgH * (imgH == maxH)
        
        max_y, max_x = np.nonzero(imgH)
        indices = [pos for pos in zip(max_y, max_x)]
        for index in indices:
            y = index[0]
            x = index[1]
            self.key_points.append(cv2.KeyPoint(x, y, 15))
            #if not is_edge_close(h, w, y, x):
            corner_list.append([y, x, imgH[y, x]])
        
        self.corner_list = corner_list
        self.corners_map = corners_map
        print(len(corner_list))
        output_image = cv2.drawKeypoints(rescale(self.image).astype('uint8'), self.key_points, self.image)
        cv2.imwrite(f_name, output_image)

if __name__ == '__main__':
    #image = cv2.imread('../checkerboard.png')
    image = cv2.imread('im1.jpg')
    #image = open_image('Yosemite1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris = Harris(image, 5, 1, 0.3)
    harris.harris_matrix('Yosemite1_result.jpg')
    harris.gradient_matrix()