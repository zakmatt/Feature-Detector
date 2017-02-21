#!/usr/bin/env python3
from controllers import *
import cv2
from math import atan2
import numpy as np
from numpy import linalg as LA
from scipy.ndimage import filters

class Harris(object):
    
    def __init__(self, image, window_size, sigma):
        self.image = image
        self.window_size = window_size
        self.sigma = sigma
    
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
                        atan2(d_y, d_x)
                        ]
    
    def harris_matrix(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.I_x, self.I_y = compute_derivatives(gray_image)
        I_xx = self.I_x ** 2
        I_yy = self.I_y ** 2
        I_xy = self.I_x * self.I_y
        
        h, w, _ = self.image.shape
        kernel = GaussianKernel(self.window_size, self.sigma)
        offset = self.window_size // 2
        corner_list = []
        
        corners_map = np.zeros((h,w))
        
        for x in range(offset, w - offset):
            for y in range(offset, h - offset):
                xx = I_xx[y - offset:y + offset + 1,
                          x - offset:x + offset + 1]
                yy = I_yy[y - offset:y + offset + 1,
                          x - offset:x + offset + 1]
                xy = I_xy[y - offset:y + offset + 1,
                          x - offset:x + offset + 1]
                
                # w * I - smooth 
                xx = np.multiply(kernel, xx)
                yy = np.multiply(kernel, yy)
                xy = np.multiply(kernel, xy)
                
                # sums
                Sxx = xx.sum()
                Syy = yy.sum()
                Sxy = xy.sum()
                
                # Find determinant and trace, use to get
                # corner strength function
                det = (Sxx * Syy) - (Sxy ** 2)
                trace = Sxx + Syy
                c = det / (trace + 1e-8) # avoiding dividing by zero
                
                if c > 1000:
                    corners_map[y - offset, x - offset] = c
                    corner_list.append([y - offset, x - offset, c])
                
        # filter three neighbours
        max_i = filters.maximum_filter(corners_map, (5,5))
        corners_map *= (corners_map == max_i)
        max_x, max_y = np.nonzero(corners_map)
        indices = [pos for pos in zip(max_x, max_y)]
        self.colored_image = color_image(self.image, indices)
        # cv2.imwrite('result.png', self.colored_image)
        self.corner_list = corner_list
        self.corners_map = corners_map
    
if __name__ == '__main__':
    #image = cv2.imread('../checkerboard.png')
    image = cv2.imread('img1.ppm')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris = Harris(image, 5, 30)
    harris.harris_matrix()
    harris.gradient_matrix()