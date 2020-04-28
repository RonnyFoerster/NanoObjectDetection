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
        plt.figure(figsize=(15,10))
        
        y_min = y_range[0]
        y_max = y_range[1]+1
        x_min = x_range[0]
        x_max = x_range[1]+1
         
        plt.imshow(image[frame,y_min:y_max, x_min:x_max], cmap = 'gray', norm=colors.PowerNorm(gamma=my_gamma))
        plt.title("Raw Image")
        plt.xlabel("x [Px]")
        plt.ylabel("y [Px]")
 
        plt.tight_layout()
        
        
    frame_slider = IntSlider(min = 1, max = max_f, description = "Frame")    
    y_range_slider = IntRangeSlider(value=[0, max_y], min=0, max=max_y, description = "ROI - y")
    x_range_slider = IntRangeSlider(value=[0, max_x], min=0, max=max_x, description = "ROI - x")
    gamma_slider = FloatSlider(min = 0.1, max = 2, step = 0.05, value = 0.5)  
        
    interact(ShowRawImage, frame = frame_slider, y_range = y_range_slider, x_range = x_range_slider, my_gamma = gamma_slider)
    
    
    
def ChooseROIParameters(rawframes_np):
    # select if ROI is used
    ApplyROI = IntSlider(min = 0, max = 1, description = "Apply ROI (0 - no, 1 - yes)")  
    
    def ChooseROI(ApplyROI):
        if ApplyROI == 0:
            print("ROI not applied")
            
        else:
            [max_f, max_y, max_x] = np.asarray(rawframes_np.shape) - 1
        
            def ShowImageROI(frame_range, y_range, x_range, my_gamma):          
                fig, axes = plt.subplots(2,1, sharex = True,figsize=(15,6))
        
                frame_min = frame_range[0]
                frame_max = frame_range[1]        
                y_min = y_range[0]
                y_max = y_range[1]+1
                x_min = x_range[0]
                x_max = x_range[1]+1
                         
                axes[0].imshow(rawframes_np[frame_min,y_min:y_max, x_min:x_max], cmap = 'gray', norm=colors.PowerNorm(gamma=my_gamma))
                axes[0].set_title("First Frame")
                axes[0].set_xlabel("x [Px]")
                axes[0].set_ylabel("y [Px]")
        
                axes[1].imshow(rawframes_np[frame_max,y_min:y_max, x_min:x_max], cmap = 'gray', norm=colors.PowerNorm(gamma=my_gamma))
                axes[1].set_title("Last Frame")
                axes[1].set_xlabel("x [Px]")
                axes[1].set_ylabel("y [Px]")        
                
                plt.tight_layout()
                        
            
            frames_range_slider = IntRangeSlider(value=[0, max_f], min=0, max=max_f, description = "ROI - frames")
            y_range_slider = IntRangeSlider(value=[0, max_y], min=0, max=max_y, description = "ROI - y")
            x_range_slider = IntRangeSlider(value=[0, max_x], min=0, max=max_x, description = "ROI - x")
            gamma_slider = FloatSlider(min = 0.1, max = 2, step = 0.05, value = 0.5)  
        
            
            interact(ShowImageROI, frame_range = frames_range_slider, y_range = y_range_slider, x_range = x_range_slider, my_gamma = gamma_slider)

    interact(ChooseROI, ApplyROI = ApplyROI)

    