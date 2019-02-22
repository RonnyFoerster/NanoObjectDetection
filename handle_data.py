# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: foersterronny
"""

# In[0]:
# coding: utf-8
"""
Analyzis of Gold-Particle data for ARHCF paper

Created on 20181001 by Stefan Weidlich (SW)
Based on previous skript of Ronny Förster (RF) and SW

Target here: Implement only routines as they'll be used in paper

Modifications:
181020, SW: Cleaning up of code
Amongst others: Python v3.5 implementation deleted. Now working with 3.6 and above only
181025, SW: Adjustment of tracking-parameters and structuring of header
--> Realized that 32-bit parameters for tracking lead to unsufficient pixel-precision for 64 bit-version
181026, SW: Taking out of log-tracking. --> Not needed

******************************************************************************
Importing neccessary libraries
"""
import numpy as np # Library for array-manipulation
import pims # pims allows image and video handling
import matplotlib.pyplot as plt # Libraries for plotting
import sys
import json
import astropy.io.fits as pyfits
import os 
from PIL import Image
import fnmatch
from skimage import io
from pdb import set_trace as bp #debugger
  
# In[2]:

def ReadJson(mypath):
    with open(mypath) as json_file:
        settings = json.load(json_file)
            
    # check if json path is written in file. otherwise put it there
    if "path_setting" in list(settings["Exp"].keys()):
        if settings["Exp"]["path_setting"] != mypath:
            sys.exit("Given Json path does not match defined path in json file! ,\
                     You might wanna delete the 'settings' row entirely from the json file.")
    else:
        settings["Exp"]["path_setting"] = mypath
    
    return settings


def GetVarOfSettings(settings, key, entry):

    if entry in list(settings[key].keys()):
        # get value
        value = settings[key][entry]
        
    else:
        print("!!! Parameter settings['%s']['%s'] not found !!!" %(key, entry))
        # see if defaul file is available
        if "DefaultParameterJsonFile" in list(settings["Exp"].keys()):
            print('!!! Default File found !!!')
            path_default = settings["Exp"]["DefaultParameterJsonFile"]
            
            settings_default = ReadJson(path_default)
            default_value = settings_default[key][entry]
            
            ReadDefault = 'invalid'
            while (ReadDefault in ["y", "n"]) == False:
                # Do until input is useful
                ReadDefault = input("Shall default value %s been used [y/n]?" %default_value)
            
            if ReadDefault == "y":
                value = default_value

            else:
                SetYourself = 'invalid'
                while (SetYourself in ["y", "n"]) == False:
                    # Do until input is useful
                    ReadDefault = input("Do you wanna set the value yourself [y/n]?")
                 
                if ReadDefault == "y":
                    value = input("Ok. Set the value of settings['%s']['%s']: " %(key, entry))
                else:
                    sys.exit("Well... you had your chances!")
                        
                
        
    return value


def SpecificValueOrSettings(try_value,settings, key, entry):
    if try_value != None:
        use_value = try_value
    else:
        # if no angle given, than use the one out of the settings
        if settings == None:
            sys.exit("Either value or settings needed")
        else:
            use_value = nd.handle_data.GetVarOfSettings(settings, key, entry)
    
    settings[key][entry] = use_value
    
    var_type = type(use_value)
    if var_type is int:
        print("%s = %d" %(entry,use_value))
    if var_type is float:
        if int(use_value) == use_value:
            print("%s = %d" %(entry,use_value))
        else:
            print("%s = %5.5f" %(entry,use_value))
    elif var_type is list:
        print("%s = " %(entry))
        print(use_value)
        
        
    return use_value, settings


"""
******************************************************************************
Reading in data and cropping image to ROI needed
"""


def ReadImages_pims(data_folder_name, data_file_extension):
    print('start reading in raw images')
    rawframes = pims.ImageSequence(data_folder_name + '\\' + '*.' + data_file_extension)
    
    return rawframes 


import nanoobject_detection as nd


def ReadData2Numpy(settings):
    data_type        = nd.handle_data.GetVarOfSettings(settings,"Exp","data_type")
    data_folder_name = nd.handle_data.GetVarOfSettings(settings,"Exp","data_folder_name")
    data_file_name   = nd.handle_data.GetVarOfSettings(settings,"Exp","data_file_name")
    
    if data_type == 'tif_series':
        if data_folder_name == 0:
            sys.exit('!!! data_folder_name required !!!')
            
        else:
            rawframes_np = nd.handle_data.ReadTiffSeries2Numpy(data_folder_name)
    
    elif data_type == 'tif_stack':
        if data_file_name == 0:
            sys.exit('!!! data_file_name required !!!')
        else:
            rawframes_np = nd.handle_data.ReadTiffStack2Numpy(data_file_name)
    
    elif data_type == 'fits':
        if data_file_name == 0:
            sys.exit('!!! data_file_name required !!!')
        else:
            rawframes_np = nd.handle_data.ReadFits2Numpy(data_file_name)   
        
    else:
        sys.exit('Data type %s' %data_type)
        
    return rawframes_np



#def ReadData2Numpy(data_type, data_file_name = 0, data_folder_name = 0):
#    if data_type == 'tif_series':
#        if data_folder_name == 0:
#            print('!!! data_folder_name required !!!')
#            quit()
#        else:
#            rawframes_np = nd.handle_data.ReadTiffSeries2Numpy(data_folder_name)
#    
#    elif data_type == 'tif_stack':
#        if data_file_name == 0:
#            print('!!! data_file_name required !!!')
#        else:
#            rawframes_np = nd.handle_data.ReadTiffStack2Numpy(data_file_name)
#    
#    elif data_type == 'fits':
#        if data_file_name == 0:
#            print('!!! data_file_name required !!!')
#        else:
#            rawframes_np = nd.handle_data.ReadFits2Numpy(data_file_name)   
#        
#    else:
#        print('unknown mode')
#        
#    return rawframes_np
#    

def ReadTiffStack2Numpy(data_file_name):
    
    print('start reading in raw images')
    rawframes_np = io.imread(data_file_name)
    
    print('finishied reading in raw images')
    
    return rawframes_np 


def ReadTiffSeries2Numpy(data_folder_name):

    
    print('start reading in raw images')
    rawframes_np = []
    for fname in os.listdir(data_folder_name):
        if fnmatch.fnmatch(fname, '*.tif'):
            im = Image.open(os.path.join(data_folder_name, fname))
            imarray = np.array(im)
            rawframes_np.append(imarray)
        else:
            print('%s is not a >tif<  file. Skipped it.'%fname)
    
    rawframes_np = np.asarray(rawframes_np) # shape = (60000,28,28)
    
    print('Be sure that tiff series in right order (0002.tif and not 2.tif (which will be sorted after 10.tif))')
    
    print('finishied reading in raw images')
    
    return rawframes_np


def ReadFits2Numpy(data_file_name):
    print('start reading in raw images')
    
    open_fits = pyfits.open(data_file_name)
    rawframes_np = open_fits[0].data
    
    print('finishied reading in raw images')
    
    return rawframes_np




def MaxROI(image):
    ROI = {'min_x' : 0,
           'max_x' : image.shape[1]-1,
           'min_y' : 0,
           'max_y' : image.shape[2]-1,
           'min_frame' : 0,
           'max_frame' : image.shape[0]-1}
    return ROI


def ChangeROI(settings,x_min = False, x_max = False, y_min = False, y_max = False,
              frame_min = False, frame_max = False):
    if x_min != False:
        settings["ROI"]['x_min'] = x_min
    if x_max != False:
        settings["ROI"]['x_max'] = x_max
    if y_min != False:
        settings["ROI"]['y_min'] = y_min
    if y_max != False:
        settings["ROI"]['y_max'] = y_max
    if frame_min != False:
        settings["ROI"]['min_frame'] = frame_min
    if frame_max != False:
        settings["ROI"]['max_frame'] = frame_max
        
    return settings


def UseROI(image, settings, x_min = None, x_max = None, y_min = None, y_max = None,
           frame_min = None, frame_max = None):
    x_min, settings = nd.handle_data.SpecificValueOrSettings(x_min,settings,"ROI",'x_min')
    x_max, settings = nd.handle_data.SpecificValueOrSettings(x_max,settings,"ROI",'x_max')
    y_min, settings = nd.handle_data.SpecificValueOrSettings(y_min,settings,"ROI",'y_min')
    y_max, settings = nd.handle_data.SpecificValueOrSettings(y_max,settings,"ROI",'y_max')
    frame_min, settings = nd.handle_data.SpecificValueOrSettings(frame_min,settings,"ROI",'frame_min')
    frame_max, settings = nd.handle_data.SpecificValueOrSettings(frame_max,settings,"ROI",'frame_max')
        
    image_ROI = image[frame_min : frame_max, x_min : x_max, y_min : y_max]
    return image_ROI, settings




def RotImages(rawframes_np, settings, Do_rotation = None, rot_angle = None):
    import scipy
    
    Do_rotation = nd.handle_data.SpecificValueOrSettings(Do_rotation,settings, "Processing", "Do_or_apply_data_rotation")
    
    if Do_rotation == True:
        rot_angle, settings = nd.handle_data.SpecificValueOrSettings(rot_angle,settings, "Processing", "rot_angle")

    
        if rawframes_np.ndim == 2:
            im_out = scipy.ndimage.interpolation.rotate(rawframes_np, angle = rot_angle, axes=(1, 0), reshape=True, output=None, order=1, mode='constant', cval=0.0, prefilter=True)
        else:
            im_out = scipy.ndimage.interpolation.rotate(rawframes_np, angle = rot_angle, axes=(1, 2), reshape=True, output=None, order=1, mode='constant', cval=0.0, prefilter=True)

        print("Rotation of rawdata: Applied with an angle of %d" %rot_angle)
        
    else:
        im_out = rawframes_np
        print("Rotation of rawdata: Not Applied")

    return im_out, settings



#def RotateImagesTiff(rawframes_np, settings):
#    
#    rot_angle = nd.handle_data.GetVarOfSettings(settings,"Processing","rot_angle")
#    
#    rot_angle = 10
#    # create new folder NAME
##    data_folder_name_save = (data_folder_name + '\\' + 'rotation_by_' + str(rot_angle) + '_degree')
#    
#    # if path not existing, than create it and fill it with rotated data
##    if os.path.isdir(data_folder_name_save) == False:
#        # make folder
##        os.makedirs(data_folder_name_save)
#    
#        # go step by step through the raw data
#        for loop_frame in range(0,len(rawframes_np)):
#            print(loop_frame)
#            # create array out of rawimage
#            np_need_rot = rawframes_np[loop_frame,:,:];
#            # create datatype 'image' out of array
#            image_need_rot = PIL.Image.fromarray(np_need_rot.astype(np.uint32));
#            # rotate the image
#            # expand means that the image size is allowed to change in order to keep all informations
#            # image DATA is rotated, but the axes stay x and y
#            # resample is the method to interpolate the rotated image points onto the standard x-y-grid 
#            # methods are: 'NEAREST', 'BILINEAR', 'BICUBIC'
#            image_after_rot = image_need_rot.rotate(rot_angle, resample=PIL.Image.BILINEAR, expand=1);
#            # save the stuff into the folder
#            image_after_rot.save(data_folder_name_save + '\\image' + str(loop_frame) + '.' + data_file_extension)
#            
##    # keep unrotated data just in case
##    frames_no_rot = rawframes
##    
##    # replace frames by rotated data, crops data
##    rawframes = pims.ImageSequence(data_folder_name_save + '\\' + '*.' + data_file_extension, process_func=crop)
##    return rot_angle
#
#




def min_rawframes(rawframes_np, display = False):
    rawframes_min = np.min(rawframes_np,axis=0)
    if display == True:
        plt.imshow(rawframes_min)
        
    return rawframes_min
    
    
def max_rawframes(rawframes_np, display = False):
    rawframes_max = np.max(rawframes_np,axis=0)
    if display == True:
        plt.imshow(rawframes_max)
        
    return rawframes_max
        

def mean_rawframes(rawframes_np, display = False):
    rawframes_mean = np.mean(rawframes_np,axis=0)
    if display == True:
        plt.imshow(rawframes_mean)
        
    return rawframes_mean
        
def percentile_rawframes(rawframes_np, percentile, display = False):
    rawframes_percentile = np.percentile(rawframes_np, percentile, axis=0)
    if display == True:
        plt.imshow(rawframes_percentile)
        
    return rawframes_percentile
    
    
def are_rawframes_saturated(rawframes_np, ignore_saturation = False):
    brightes_pixel = np.max(rawframes_np)
    
    # is it a multiple of 2^x ... if so it sounds saturated
    hot_pixel = float(np.log2(brightes_pixel + 1)).is_integer()
    if hot_pixel == True:
        if ignore_saturation == False:
            sys.exit("Your Data seems saturated")
        else:
            print("Your Data seems saturated - but you dont care...")
 
    
def total_intensity(rawframes_np, display = False):
    # intensity in each frame
    tot_intensity = np.sum(rawframes_np,axis=(1,2))
    
    # intensity in each frame relative to the others
    rel_intensity = tot_intensity / np.mean(tot_intensity)
    
    if display == True:
        plt.plot(rel_intensity)
    
        
    return tot_intensity, rel_intensity


def NormImage(image):
    image = image - np.min(image)
    image = image / np.max(image)
    
    return image


def DispWithGamma(image, gamma = 0.5, display = False):
    image = NormImage(image)
    print(gamma)
    image = image ** gamma
    return image
       

def LogData(rawframes):
    # Stefan loves the log
    rawframes_log=np.log(rawframes)
    rawframes_log_median=np.median(rawframes_log)
    rawframes_log[rawframes_log==-np.inf]=rawframes_log_median
    rawframes_log[rawframes_log<0]=0     
            
            
    return rawframes_log
            
## Crop image to the region of interest only (if needed re-define cropping parameters above):
#def crop(img): 
#    x_min = x_min_global
#    x_max = x_max_global
#    y_min = y_min_global
#    y_max = y_max_global 
#    #return img[y_min:y_max,x_min:x_max,0] # ATTENTION: This form is used for images that are stored as 2d-data (*tif)!
#    return img[y_min:y_max,x_min:x_max] # ATTENTION: Use this form if not using 2d-data (e.g. *.bmp)!
##rawframes = pims.ImageSequence(data_folder_name + '\\' + '*.' + data_file_extension, process_func=crop)
## ATTENTION: Data get's cut into a smaller part here above. When data should be rotated, it's better to do this first and then cut
#"""
#Remark
#Depending on how data is stored, the return needs an additional "0".
#The procedure here can be used to crop the image to a smaller slice first. 
#Yet, when rotating the image lateron, it's better not to do this, but to crop after rotation.
#"""


