# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:17:55 2020

@author: foersterronny
"""
from ipywidgets import IntSlider, IntRangeSlider, FloatLogSlider, FloatSlider
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt

# In[Functions]

def Show3dImage(image):
    [max_f, max_y, max_x] = np.asarray(image.shape) - 1
    
    def ShowRawImage(frame, y_range, x_range, my_gamma):  
        
        fig, axes = plt.plot(figsize=(15,10))
        
        y_min = y_range[0]
        y_max = y_range[1]+1
        x_min = x_range[0]
        x_max = x_range[1]+1
         
        axes.imshow(image[frame,y_min:y_max, x_min:x_max], cmap = 'gray', norm=colors.PowerNorm(gamma=my_gamma))
        axes.set_title("Raw Image")
        axes.set_ylabel("y [Px]")
 
        fig.tight_layout()
        
        
    frame_slider = IntSlider(min = 1, max = max_f, description = "Frame")    
    y_range_slider = IntRangeSlider(value=[0, max_y], min=0, max=max_y, description = "ROI - y")
    x_range_slider = IntRangeSlider(value=[0, max_x], min=0, max=max_x, description = "ROI - x")
    gamma_slider = FloatSlider(min = 0.1, max = 2, step = 0.05, value = 1)  
        
    interact(ShowRawImage, frame = frame_slider, y_range = y_range_slider, x_range = x_range_slider, my_gamma = gamma_slider)