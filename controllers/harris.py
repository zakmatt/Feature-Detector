#!/usr/bin/env python3
from controllers import *
import cv2
import numpy as np

class Harris(object):
    
    def __init__(self, image, window_size, sigma):
        self.image = image
        self.window_size = window_size
        self.sigma = sigma
    
    def harris_matrix(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        I_x, I_y = compute_derivatives(gray_image)
        I_xx = I_x ** 2
        I_yy = I_y ** 2
        I_xy = I_x * I_y
        
        h, w, _ = image.shape
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
        cv2.imwrite('result.png', self.colored_image)
        self.corner_list = corner_list
        self.corners_map = corners_map
        # return corner_list, colored_image, tmp
    
if __name__ == '__main__':
    image = cv2.imread('../checkerboard.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris = Harris(image, 5, 30)
    harris.harris_matrix()
    #c, colored_image, tmp = harris_matrix(image,
    #                  5,
    #                  30)