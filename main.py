#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 18:10:08 2017

@author: Matthew Zak
"""

from controllers.controllers import *
#import cv2
from controllers.harris import Harris
from controllers.feature_description import *
from controllers.feature_matching import *

if __name__ == '__main__':
    
    # 1 harris corner detection, normal
    
    image = controllers.open_image('images/normal/Yosemite1.jpg')
    # Threshold 0.3
    harris = Harris(image, 5, 1, 0.3)
    harris.harris_matrix_adaptive()
    output_image = harris.output_image
    save_image(output_image, 'output/harris_corner_detection/harris_normal_03.jpg')
    print('image saved')
    
    # Threshold 0.8
    harris = Harris(image, 5, 1, 0.8)
    harris.harris_matrix_adaptive()
    output_image = harris.output_image
    save_image(output_image, 'output/harris_corner_detection/harris_normal_08.jpg')
    print('image saved')
    
    # Matching 
    # normal image
    
    image_1 = open_image('images/normal/Yosemite1.jpg')
    
    # normal
    image_2 = open_image('images/normal/Yosemite2.jpg')
    description_1, corners_1, key_points_1 = get_image_description(
            image_1, 5, 1, 0.3
            )
    description_2, corners_2, key_points_2 = get_image_description(
            image_2, 5, 1, 0.3
            )
    
    matches = match_features(description_1, 
                             description_2, 
                             0.8)
    
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:10]
    
    im3 = cv2.drawMatches(image_1, 
                          key_points_1,
                          image_2, 
                          key_points_2, 
                          matches,
                          None,
                          flags = 2)
    save_image(im3, 'output/normal_matches/normal_match.jpg')
    
    
    # normal, changed luminosity
    image_2 = open_image('images/normal_luminosity/Yosemite2.jpg')
    description_1, corners_1, key_points_1 = get_image_description(
            image_1, 5, 1, 0.3
            )
    description_2, corners_2, key_points_2 = get_image_description(
            image_2, 5, 1, 0.3
            )
    
    matches = match_features(description_1, 
                             description_2, 
                             0.8)
    
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:10]
    
    im3 = cv2.drawMatches(image_1, 
                          key_points_1,
                          image_2, 
                          key_points_2, 
                          matches,
                          None,
                          flags = 2)
    save_image(im3, 'output/normal_luminosity/normal_luminosity.jpg')
    
    # normal, rescaled
    image_2 = open_image('images/normal_rescaled/Yosemite2.jpg')
    description_1, corners_1, key_points_1 = get_image_description(
            image_1, 5, 1, 0.3
            )
    description_2, corners_2, key_points_2 = get_image_description(
            image_2, 5, 1, 0.3
            )
    
    matches = match_features(description_1, 
                             description_2, 
                             0.8)
    
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:10]
    
    im3 = cv2.drawMatches(image_1, 
                          key_points_1,
                          image_2, 
                          key_points_2, 
                          matches,
                          None,
                          flags = 2)
    save_image(im3, 'output/normal_rescaled/normal_rescaled.jpg')
    
    # normal, rescaled, luminosity
    image_2 = open_image('images/normal_rescaled_liminosity/Yosemite2.jpg')
    description_1, corners_1, key_points_1 = get_image_description(
            image_1, 5, 1, 0.3
            )
    description_2, corners_2, key_points_2 = get_image_description(
            image_2, 5, 1, 0.3
            )
    
    matches = match_features(description_1, 
                             description_2, 
                             0.8)
    
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:10]
    
    im3 = cv2.drawMatches(image_1, 
                          key_points_1,
                          image_2, 
                          key_points_2, 
                          matches,
                          None,
                          flags = 2)
    save_image(im3, 'output/normal_rescaled_liminosity/normal_rescaled_luminosity.jpg')
    
    
    # rotated image
    
    image_1 = open_image('images/rotated/img1.ppm')
    
    # rotted
    image_2 = open_image('images/rotated/img2.ppm')
    description_1, corners_1, key_points_1 = get_image_description(
            image_1, 5, 1, 0.3
            )
    description_2, corners_2, key_points_2 = get_image_description(
            image_2, 5, 1, 0.3
            )
    
    matches = match_features(description_1, 
                             description_2, 
                             0.8)
    
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:10]
    
    im3 = cv2.drawMatches(image_1, 
                          key_points_1,
                          image_2, 
                          key_points_2, 
                          matches,
                          None,
                          flags = 2)
    save_image(im3, 'output/rotated/rotated.jpg')
    
    
    # rotated, changed luminosity
    image_2 = open_image('images/rotated_luminosity/img2.ppm')
    description_1, corners_1, key_points_1 = get_image_description(
            image_1, 5, 1, 0.3
            )
    description_2, corners_2, key_points_2 = get_image_description(
            image_2, 5, 1, 0.3
            )
    
    matches = match_features(description_1, 
                             description_2, 
                             0.8)
    
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:10]
    
    im3 = cv2.drawMatches(image_1, 
                          key_points_1,
                          image_2, 
                          key_points_2, 
                          matches,
                          None,
                          flags = 2)
    save_image(im3, 'output/rotated_luminosity/rotated_luminosity.jpg')
    
    # rotated, rescaled
    image_2 = open_image('images/roteted_rescaled/img2.ppm')
    description_1, corners_1, key_points_1 = get_image_description(
            image_1, 5, 1, 0.3
            )
    description_2, corners_2, key_points_2 = get_image_description(
            image_2, 5, 1, 0.3
            )
    
    matches = match_features(description_1, 
                             description_2, 
                             0.8)
    
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:10]
    
    im3 = cv2.drawMatches(image_1, 
                          key_points_1,
                          image_2, 
                          key_points_2, 
                          matches,
                          None,
                          flags = 2)
    save_image(im3, 'output/roteted_rescaled/rotated_rescaled.jpg')
    
    # rotated, rescaled, luminosity
    image_2 = open_image('images/rotated_rescaled_luminosity/img2.ppm')
    description_1, corners_1, key_points_1 = get_image_description(
            image_1, 5, 1, 0.3
            )
    description_2, corners_2, key_points_2 = get_image_description(
            image_2, 5, 1, 0.3
            )
    
    matches = match_features(description_1, 
                             description_2, 
                             0.8)
    
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:10]
    
    im3 = cv2.drawMatches(image_1, 
                          key_points_1,
                          image_2, 
                          key_points_2, 
                          matches,
                          None,
                          flags = 2)
    save_image(im3, 'output/rotated_rescaled_luminosity/rotated_rescaled_luminosity.jpg')