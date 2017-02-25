#!/usr/bin/env python3
import cv2
from controllers import *
from harris import Harris
import numpy as np

def get_description(harris):
    
    def create_histogram(neighbour_gradient_matrix):
        # vector of 8 bins
        histogram = np.zeros(8)
        #d_x,
        #d_y,
        #LA.norm([d_x, d_y]),
        #cv2.fastAtan2(d_y, d_x)
        h, w, _ = neighbour_gradient_matrix.shape

        for y in range(h):
            for x in range(w):
                element = neighbour_gradient_matrix[y, x]
                bin_position = int(element[3] / 45) % 8
                histogram[bin_position] += element[2]
        return histogram
        
    corner_list = harris.corner_list
    gradient_matrix = harris.gradient_matrix
    all_neighbourhoods = []
    for corner in corner_list:
        y, x, c = corner
        nearest_neighbourhood = []
        # take neighbours
        # y and x axes
        # 16 by 16 window
        for p in range(-8, 8, 4):
            for q in range(-8, 8, 4):
                nearest_neighbourhood.append(
                        create_histogram(
                                gradient_matrix[
                                        y + p:y + p + 4,
                                        x + q:x + q + 4
                                        ]
                                )
                        )
        all_neighbourhoods.append(nearest_neighbourhood)
    return np.array(all_neighbourhoods)

if __name__ == '__main__':
    image = open_image('img1.ppm')
    #image = open_image('Yosemite1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris = Harris(image, 5, 1, 0.3)
    harris.harris_matrix('cdsds.jpg')
    harris.gradient_matrix()
    all_neighbourhoods = get_description(harris)
    #print(all_neighbourhoods)
    print(all_neighbourhoods.shape)
    