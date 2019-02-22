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
import NanoObjectDetection as nd

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
#    bp()
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





def min_rawframes(rawframes_np, display = False):
    import NanoObjectDetection as nd
    rawframes_min = np.min(rawframes_np,axis=0)
    
    if display == True:
        title = "Background image"
        xlabel = "long. Position [Px]"
        ylabel = "trans. Position [Px]"
        nd.visualize.Plot2DImage(rawframes_min, title, xlabel, ylabel)

        
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
    import NanoObjectDetection as nd
    # intensity in each frame
    tot_intensity = np.sum(rawframes_np,axis=(1,2))
    
    # intensity in each frame relative to the others
    rel_intensity = tot_intensity / np.mean(tot_intensity)
    
    if display == True:
        nd.visualize.Plot1DPlot(rel_intensity, "Laser Fluctuations", "Frame", "Relative Laser Intensity")
    
        
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


