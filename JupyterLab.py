# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:17:55 2020

@author: foersterronny
"""
from ipywidgets import IntSlider, IntRangeSlider, FloatSlider, Dropdown, FloatLogSlider, BoundedIntText, IntText
from ipywidgets import interact, interactive


import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt

import NanoObjectDetection as nd

from pdb import set_trace as bp #debugger
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
        
        
    frame_slider = IntSlider(min = 1, max = max_f, step = 1, description = "Frame")    
    y_range_slider = IntRangeSlider(value=[0, max_y], min=0, max=max_y, step = 1, description = "ROI - y")
    x_range_slider = IntRangeSlider(value=[0, max_x], min=0, max=max_x, step = 1, description = "ROI - x")
    gamma_slider = FloatSlider(min = 0.1, max = 2, step = 0.05, value = 0.5)  


        
    interact(ShowRawImage, frame = frame_slider, y_range = y_range_slider, x_range = x_range_slider, my_gamma = gamma_slider)
    

   
    
def ChooseROIParameters(rawframes_np, ParameterJsonFile):
    #read in settings
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    # select if ROI is used
    ApplyROI = IntSlider(min = 0, max = 1, value = settings["ROI"]["Apply"], description = "Apply ROI (0 - no, 1 - yes)")  
    
    def ChooseROI(ApplyROI):
        settings = nd.handle_data.ReadJson(ParameterJsonFile)
        settings["ROI"]["Apply"] = ApplyROI
        
        if ApplyROI == 0:
            print("ROI not applied")
            
        else:
            [max_f, max_y, max_x] = np.asarray(rawframes_np.shape) - 1

            def ShowImageROI(frame_range, y_range, x_range, my_gamma):  
                settings = nd.handle_data.ReadJson(ParameterJsonFile)
                fig, axes = plt.subplots(2,1, sharex = True,figsize=(15,6))
        
                frame_min = frame_range[0]
                frame_max = frame_range[1]        
                y_min = y_range[0]
                y_max = y_range[1]
                x_min = x_range[0]
                x_max = x_range[1]

                settings["ROI"]["frame_min"] = frame_min
                settings["ROI"]["frame_max"] = frame_max
                settings["ROI"]["y_min"] = y_min
                settings["ROI"]["y_max"] = y_max
                settings["ROI"]["x_min"] = x_min
                settings["ROI"]["x_max"] = x_max
                                         
                nd.handle_data.WriteJson(ParameterJsonFile, settings)
                
                axes[0].imshow(rawframes_np[frame_min,y_min:y_max+1, x_min:x_max+1], cmap = 'gray', norm=colors.PowerNorm(gamma=my_gamma))
                axes[0].set_title("First Frame")
                axes[0].set_xlabel("x [Px]")
                axes[0].set_ylabel("y [Px]")
        
                axes[1].imshow(rawframes_np[frame_max,y_min:y_max+1, x_min:x_max+1], cmap = 'gray', norm=colors.PowerNorm(gamma=my_gamma))
                axes[1].set_title("Last Frame")
                axes[1].set_xlabel("x [Px]")
                axes[1].set_ylabel("y [Px]")       
                
                plt.tight_layout()                       
            
            #insert the starting values from the json file
            frames_range_slider = IntRangeSlider(value=[settings["ROI"]["frame_min"], settings["ROI"]["frame_max"]], min=0, max=max_f, step = 10, description = "ROI - frames")
            
            y_range_slider = IntRangeSlider(value=[settings["ROI"]["y_min"], settings["ROI"]["y_max"]], min=0, max=max_y, step = 10, description = "ROI - y")
            
            x_range_slider = IntRangeSlider(value=[settings["ROI"]["x_min"], settings["ROI"]["x_max"]], min=0, max=max_x, step = 10, description = "ROI - x")
            
            gamma_slider = FloatSlider(min = 0.1, max = 2, step = 0.05, value = 0.5)  
        
            
            interact(ShowImageROI, frame_range = frames_range_slider, y_range = y_range_slider, x_range = x_range_slider, my_gamma = gamma_slider)

        nd.handle_data.WriteJson(ParameterJsonFile, settings)

    interact(ChooseROI, ApplyROI = ApplyROI)




def ChoosePreProcessingParameters(rawframes_np, ParameterJsonFile):
    #read in settings
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    # LASER FLUCTUATIONS
    # select if Laser Fluctuations are applied
    process_laser_fluctuations = IntSlider(min = 0, max = 1, \
                         value = settings["PreProcessing"]["Remove_Laserfluctuation"],\
                         description = "Correct Laser fluctuations (0 - no, 1 - yes)")  

    def RemoveLaserFluctuations(process_laser_fluctuations):
        settings = nd.handle_data.ReadJson(ParameterJsonFile)
        settings["PreProcessing"]["Remove_Laserfluctuation"] = process_laser_fluctuations

        if process_laser_fluctuations == 0:
            print("Laser Fluctuations not corrected")
        else:
            settings["Plot"]['Laserfluctuation_Show'] = True
            nd.PreProcessing.RemoveLaserfluctuation(rawframes_np, settings)    

        nd.handle_data.WriteJson(ParameterJsonFile, settings)

    interact(RemoveLaserFluctuations, process_laser_fluctuations = process_laser_fluctuations)



    # CAMERA OFFSET
    process_camera_offset = IntSlider(min = 0, max = 1, \
                         value = settings["PreProcessing"]["Remove_CameraOffset"],\
                         description = "Correct camera offset (0 - no, 1 - yes)")  

    def RemoveCameraOffset(process_camera_offset):
        settings = nd.handle_data.ReadJson(ParameterJsonFile)
        settings["PreProcessing"]["Remove_CameraOffset"] = process_camera_offset

        if process_camera_offset == 0:
            print("Camera Offset not corrected")
        else:
            nd.PreProcessing.SubtractCameraOffset(rawframes_np, settings)    

        nd.handle_data.WriteJson(ParameterJsonFile, settings)

    interact(RemoveCameraOffset, process_camera_offset = process_camera_offset)



    # STATIC BACKGROUND
    process_static_background = IntSlider(min = 0, max = 1, \
                         value = settings["PreProcessing"]["Remove_StaticBackground"],\
                         description = "Correct static background (0 - no, 1 - yes)")  

    def RemoveStaticBackground(process_static_background):
        settings = nd.handle_data.ReadJson(ParameterJsonFile)
        settings["PreProcessing"]["Remove_CameraOffset"] = process_static_background

        if process_static_background == 0:
            print("Static Background not corrected")
        else:
            settings["Plot"]['Background_Show'] = True
            nd.PreProcessing.Remove_StaticBackground(rawframes_np, settings)    

        nd.handle_data.WriteJson(ParameterJsonFile, settings)

    interact(RemoveStaticBackground, process_static_background = process_static_background)



    print("RollingPercentilFilter not inserted yet")


    
    print("Clipping negative values not inserted yet. Clipping is bad")



    # ENHANCE SNR BY CONVOLVING WITH PSF
    EnhanceSNR_Slider = IntSlider(min = 0, max = 1, \
                         value = settings["PreProcessing"]["EnhanceSNR"],\
                         description = "Enhance SNR by convolving with the PSF (0 - no, 1 - yes)")  

    KernelSize = settings["PreProcessing"]["KernelSize"]
    if KernelSize == 'auto':
            KernelSize = 0    
    
    KernelSize_Slider = FloatSlider(min = 0, max = 10, \
                     value = KernelSize,\
                     description = "Kernelsize (0 - auto)")  


    def ConvolveWithPSF(EnhanceSNR, KernelSize):
        settings = nd.handle_data.ReadJson(ParameterJsonFile)
        settings["PreProcessing"]["EnhanceSNR"] = EnhanceSNR
        
        if KernelSize == 0:
            KernelSize = 'auto'
            
        settings["PreProcessing"]["KernelSize"] = KernelSize

        if EnhanceSNR == 0:
            print("SNR not enhanced by a convolution with the PSF")
        else:
            settings["Plot"]['Background_Show'] = True
            nd.PreProcessing.ConvolveWithPSF(rawframes_np[0,:,:], settings,  ShowFirstFrame = True)    

     
        nd.handle_data.WriteJson(ParameterJsonFile, settings)
        
    interact(ConvolveWithPSF, EnhanceSNR = EnhanceSNR_Slider, KernelSize = KernelSize_Slider)



    print("Rotating the image is not inserted yet. Rotate your camera if that is a problem.")
    


def ChooseFindObjParameters(rawframes_pre, ParameterJsonFile):
    #read in settings
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    # SEPARATION DISTANCE
    # select if help is required
    help_sep_distance = Dropdown(
            options=['0', 'auto'],
            value=settings["Help"]["Separation"],
            description='Help separation distance')

    def ChooseSepDistance_Mode(mode):
        
        settings["Help"]["Separation"] = mode
        
        if mode == "auto":
            Low_Diam_slider = IntSlider(min = 1, max = 100, step = 1, \
                                                value = 10, 
                                                description = "Guess lowest diameter [nm]")    
            
            def CalcSepDistance(Low_Diam):
                Min_Separation, Max_displacement = \
                nd.ParameterEstimation.FindMaxDisplacementTrackpy(ParameterJsonFile, GuessLowestDiameter_nm = Low_Diam)               
                
                settings["Find"]["Separation data"] = Min_Separation
                settings["Link"]["Max displacement"] = Max_displacement
                nd.handle_data.WriteJson(ParameterJsonFile, settings)
        
            interact(CalcSepDistance, Low_Diam = Low_Diam_slider)
        
        else:
            Min_Separation_slider = FloatSlider(min = 0, max = 100, step = 0.5, \
                                                value = settings["Find"]["Separation data"],\
                                                description = "Separation distance [Px]")    
    
            Max_displacement_slider = FloatSlider(min = 0, max = 50, step = 0.5, \
                                                  value = settings["Link"]["Max displacement"],\
                                                  description = "Maximal Displacement [Px]") 
            
            
            def ChooseSepDistance_Value(Min_Separation, Max_displacement):
                settings["Find"]["Separation data"] = Min_Separation
                settings["Link"]["Max displacement"] = Max_displacement
                
                nd.handle_data.WriteJson(ParameterJsonFile, settings)
        
            interact(ChooseSepDistance_Value, \
                     Min_Separation = Min_Separation_slider, Max_displacement = Max_displacement_slider)


        nd.handle_data.WriteJson(ParameterJsonFile, settings)


    interact(ChooseSepDistance_Mode, mode = help_sep_distance)





    # BEAD DIAMETER
    # select if help is required
    help_diameter = Dropdown(
            options=['0', 'manual', 'auto'],
            value=settings["Help"]["Bead size"],
            description='Help bead diameter')

    diameter_slider = IntSlider(min = 1, max = 31, step = 2, \
                         value = settings["Find"]["Estimated particle size"],\
                         description = "Diameter of bead [Px]")      
    
    def OptimizeBeamDiameter(mode, diameter):

        settings = nd.handle_data.ReadJson(ParameterJsonFile)
        settings["Find"]["Estimated particle size"] = diameter
        settings["Help"]["Bead size"] = mode
                
        if mode  == "manual":
            settings["Find"]["Estimated particle size"] = \
            nd.AdjustSettings.SpotSize_manual(rawframes_pre, settings, AutoIteration = False)
            
        elif settings["Help"]["Bead size"] == "auto":
            settings["Find"]["Estimated particle size"] = nd.AdjustSettings.SpotSize_auto(settings)
            
        else:
            print("Bead size not adjusted. Use 'manual' or 'auto' if you want to do it.")

        nd.handle_data.WriteJson(ParameterJsonFile, settings)

    interact(OptimizeBeamDiameter, mode = help_diameter, diameter = diameter_slider)



    # MINMASS
    # optimize minmass to identify particle
    help_minmass = Dropdown(
        options=['0', 'manual', 'auto'],
        value=settings["Help"]["Bead brightness"],
        description='Help bead minmass')
    

#    minmass_slider = FloatLogSlider(min = 1, max = 4, step = 0.1, \
    minmass_slider = IntText(value = settings["Find"]["Minimal bead brightness"],\
                         description = "Minimal bead brightness", step = 50)      
    
    frame_min = 1
    frame_max = settings["ROI"]["frame_max"] - settings["ROI"]["frame_min"]
    
    #IntText
    frame_slider = IntText(value = 0,\
                         description = "Frame", step = 1, \
                         min = frame_min, max = frame_max)  
    
    def OptimizeMinmass(mode, minmass, frame):

        settings = nd.handle_data.ReadJson(ParameterJsonFile)
        settings["Find"]["Minimal bead brightness"] = minmass
        settings["Help"]["Bead brightness"] = mode
          
        nd.handle_data.WriteJson(ParameterJsonFile, settings)
        
        if mode  == "manual":
            nd.AdjustSettings.FindSpot_manual(rawframes_pre[frame:frame+1,:,:], ParameterJsonFile, ExternalSlider = True)
            
        elif settings["Help"]["Bead size"] == "auto":
            minmass, num_particles_trackpy = nd.ParameterEstimation.EstimateMinmassMain(rawframes_pre, settings)
            settings["Find"]["Minimal bead brightness"] = minmass
            
        else:
            print("Bead size not adjusted. Use 'manual' or 'auto' if you want to do it.")

        
        nd.handle_data.WriteJson(ParameterJsonFile, settings)

    interact(OptimizeMinmass, mode = help_minmass, minmass = minmass_slider, frame = frame_slider)
    
    
    
    
def ShowQGrid(qgrid):
    qgrid



def Test():
    print("Run test file...")

    from ipywidgets import HBox, Label

#    KernelSizeSlider = FloatSlider(min = 0, max = 10, \
#                     value = 1)
#    description = Label("Kernelsize (0 - auto)")
#    
#    HBox((description, KernelSizeSlider))
    
    label = Label('A really long description')
    KernelSizeSlider = FloatSlider(min = 0, max = 10, value = 1)
    HBox([label, KernelSizeSlider])
        
#    frame_slider = IntSlider(min = 1, max = 10, description = "Frame")   
    
    def printX(x):
        print(x)
    
#    interact(printX, x = frame_slider)
    interact(printX, x = KernelSizeSlider)
    
    label = Label('A really long description')
    my_slider = IntSlider()
    HBox([label, my_slider])
    
    print("... Test file finished")