import numpy as np
import cv2

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.preprocess import binarize

class Mask:
    __slots__ = ('mask', 'palette')
    def __init__(self, wsi_shape):
        self.mask = np.zeros((wsi_shape), dtype=np.uint8)
        self.palette = {'red':(250, 0, 0), 
                        'green':(0, 250, 0), 
                        'blue':(0, 0, 250), 
                        'yellow':(250, 250, 0)}

    def draw_rect(self, **kwargs):
        # start_point, end_point, pattern_color:str
        start_point, end_point, pattern_color = kwargs['start_point'], kwargs['end_point'], kwargs['pattern_color'] 
        try:
            self.mask = cv2.rectangle(self.mask, start_point, end_point, self.palette[pattern_color], -1) # fill
        except KeyError as e:
            print('Not exist Key', e)
    
    def remove_unecessary_region(self, img):
        binary = binarize(img)
        
        # foreground
        binary = cv2.merge((binary,binary,binary))    
        foreground = cv2.bitwise_and(img,binary)
        
        # background
        background = foreground.copy()
        background[background!=0] = 255 # pixel value
        background=cv2.bitwise_not(background)
        
        return foreground + background

    def overlap(self, img, alpha=1, beta=0.3, gamma=0):
        overlap_img = cv2.addWeighted(img, alpha, self.mask, beta, gamma)
        overlap_img = self.remove_unecessary_region(overlap_img)
        
        return overlap_img