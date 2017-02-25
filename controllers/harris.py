#!/usr/bin/env python3
from controllers import *
import cv2
import numpy as np
from numpy import linalg as LA
from scipy.ndimage import filters

class Harris(object):
    
    def __init__(self, image, window_size, sigma, threshold):
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
    
    def harris_matrix(self):
        
        def is_edge_close(h, w, y, x):
            is_edge_close = False
            if (y - 16) <= 0 or (y + 16) >= h:
                is_edge_close =  True
            if (x - 16) <= 0 or (x + 16) >= w:
                is_edge_close =  True
            
            return is_edge_close
        
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.I_x, self.I_y = compute_derivatives(gray_image)
        I_xx = self.I_x ** 2
        I_yy = self.I_y ** 2
        I_xy = self.I_x * self.I_y
        
        h, w, _ = self.image.shape
        kernel = GaussianKernel(self.window_size, self.sigma)
        offset = self.window_size // 2
        corner_list = []
        self.key_points = []
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
                # make sure we don't color points at the edges
                
        '''
                if c > self.threshold and not is_edge_close(h, w, y, x):
                    #corners_map[y - offset, x - offset] = c
                    #corner_list.append([y - offset, x - offset, c])
                    self.key_points.append(cv2.KeyPoint(x, y, 15))
                    corners_map[y, x] = c
                    corner_list.append([y, x, c])
                
        # filter three neighbours
        max_i = filters.maximum_filter(corners_map, (50,50))
        corners_map *= (corners_map == max_i)
        max_y, max_x = np.nonzero(corners_map)
        #indices = [pos for pos in zip(max_y, max_x)]
        #self.colored_image = color_image(self.image, indices)
        output_image = cv2.drawKeypoints(self.image, self.key_points, self.image)
        cv2.imwrite('result.png', output_image)
        self.corner_list = corner_list
        self.corners_map = corners_map
        '''

if __name__ == '__main__':
    #image = cv2.imread('../checkerboard.png')
    image = cv2.imread('bicycle.bmp')
    #image = cv2. imread('img1.ppm')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris = Harris(image, 5, 5, 3000)
    harris.harris_matrix()
    harris.gradient_matrix()