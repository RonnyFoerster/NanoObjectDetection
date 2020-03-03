# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: Ronny Förster und Stefan Weidlich

This module take care about reading, writing, changing and simple analysis of the rawdata
"""

# In[0]: Importing neccessary libraries

import numpy as np # Library for array-manipulation
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

def ReadJson(mypath, CreateNew = False):
    """ read the json parameter file into a dictionary
    
    mypath: path to the json file
    """
    changed_something = False


    with open(mypath) as json_file:
        settings = json.load(json_file)
    
    
    # check if json path is written in file. otherwise put it there
    if "json" in list(settings["File"].keys()):
        
        # compare given and saved json path
        # lower is needed to convert Lib to lib, which is identical in the explorer
        comp1 = settings["File"]["json"].lower()
        comp2 = mypath.lower()
        if comp1 != comp2:
            if CreateNew == False:
                print(comp1)
                print(comp2)
                sys.exit("Given Json path does not match defined path in json file! ,\
                         You might wanna delete the 'settings' row entirely from the json file.")
            else:
                settings["File"]["json"] = mypath
                changed_something = True
    else:
        settings["File"]["json"] = mypath
    
    # check if some keys are missing.
    mypath_default = settings["File"]["DefaultParameterJsonFile"]
    if mypath_default == "default":
        mypath_default = os.path.dirname(nd.__file__) + "\\default_json.json"
    
    try:    
        with open(mypath_default) as json_file:
            settings_default = json.load(json_file)
    except:
        sys.exit("Default Json file probably not found. You can insert default at the key DefaultParameterJsonFile.")
       
    
    #go through all the main keys in defauls
    for loop_key in settings_default.keys():
        # check if the current key is in
        if loop_key in settings.keys():
            # if so check all subkeys
            for loop_subkey in settings_default[loop_key].keys():
                if (loop_subkey in settings[loop_key].keys()) == False:
                    print("Insert defaul key: %s with subkey: %s" %(loop_key,loop_subkey))
                    settings[loop_key][loop_subkey] = settings_default[loop_key][loop_subkey]
                    changed_something = True
        
        
        # if not copy the entire defaul key
        else:
            print("Insert defaul key: %s" %(loop_key))
            settings[loop_key] = settings_default[loop_key]
            changed_something = True
     
    
    # set SaveFolder in case of auto     
    if settings["Plot"]["SaveFolder"] == "auto":
#        settings["Plot"]["SaveFolder"] = settings["File"]["data_folder_name"] + "\\analysis"
        settings["Plot"]["SaveFolder"] = os.path.dirname(settings["File"]["json"]) + "\\analysis"
        changed_something = True
    
    # check if saving folders are valid    
    my_path = settings["Plot"]["SaveFolder"]
    
    invalid, my_path = CheckIfFolderGeneratable(my_path)
     
    if invalid == True:
        settings["Plot"]["SaveFolder"] = my_path
        changed_something = True
    
    # set SaveProperties in case of auto  
    if settings["Plot"]["SaveProperties"] == "auto":
        settings["Plot"]["SaveProperties"] = settings["Plot"]["SaveFolder"]
        changed_something = True
    
    # check if saving folders are valid
    my_path = settings["Plot"]["SaveProperties"]
    invalid, my_path = CheckIfFolderGeneratable(my_path)
     
    if invalid == True:
        settings["Plot"]["SaveProperties"] = my_path
        changed_something = True
         
            
    if changed_something == True:
        WriteJson(mypath, settings)    
    
    return settings



def CheckIfFolderGeneratable(my_path):
    invalid = False
    try:
        os.stat(my_path)
    except:
        try:
            os.mkdir(my_path)
        except:
            my_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
            invalid = True
            print("Path not accessible. Write on desktop.")

    return invalid, my_path



def WriteJson(mypath, settings):
    """ write the current settings to a json file
    
    mypath: path to the json file
    settings
    """
    with open(mypath, 'w') as outfile:
        json.dump(settings, outfile, indent = 5)



def ARHCF_HexToDiameter(side_length):
    """ calculates the inner and the outer diameter of hexagon
    
    side_length: list or array with the six side lengths
    """
    # to understand: paint hexagon and look at the 6 subtriangles of equal side length
    side_length = np.asarray(side_length)
    radius_outer = np.mean(side_length)
    radius_inner = np.sqrt(3)/2 * radius_outer
    
    diameter_outer = 2*radius_outer
    diameter_inner = 2*radius_inner
    
    return diameter_inner, diameter_outer



def GetViscosity(temperature = 295.15, solvent = "Water"):
    import CoolProp as CP
    my_visc = CP.CoolProp.Props('V','T',temperature,'P',101.3,solvent)
    
    # units
    my_visc = my_visc * 1e-12
    
    return my_visc



def GetTrajLengthAndParticleNumber(t):
    """ extract ID and trajectory length of the particle with the longest trajectory
    
    t: trajectory DataFrame
    """
 
    particle_list = list(t.particle.drop_duplicates())
    
    longest_particle = 0
    longest_traj = 0
    for test_particle in particle_list:
        test_t = t[t["particle"] == test_particle]
        traj_length = np.max(test_t["frame"]) - np.min(test_t["frame"]) 
        
        if traj_length > longest_traj:
            longest_traj = int(traj_length)
            longest_particle = int(test_particle)
            print(longest_particle, longest_traj)
    
    return longest_particle, longest_traj



def Get_min_max_round(array_in, decade):
    """ get the minimum and maximum of an array rounded to the next decade
    
    This can be useful to get limits of a plot nicely (does not end at 37.9 but at 40)
    decade: precision of np.round
    e.g.: decade = -2 --> round to 100, 200, 300, ...
    e.g.: decade = +3 --> round to 3.141
    """
 
    if decade < 0:
        sys.exit("decade must be non negative")
    else:
        my_min = np.round(np.min(array_in) - 5*np.power(10,decade-1),- decade)
        my_max = np.round(np.max(array_in) + 5*np.power(10,decade-1),- decade)
        
        min_max = [my_min, my_max]
                  
    return min_max



def GetVarOfSettings(settings, key, entry):
    """ read the variable inside a dictonary
    
    settings: dict
    key: type of setting
    entry: variable name
    old function - should no be in use anymore
    """
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
    """ check if a specific value is given. If not the one out of the settings is used
    
    Arg:
        try_value: if existing, that one is used an written in the settings
    """
    
    
#    print("key: ", key)
#    print("entry: ", entry)
#    bp()
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



def ReadData2Numpy(ParameterJsonFile):
    """ read the images in
    
    distinguishes between:
    # tif_series
    # tif_stack
    # fits
    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    
    DoSimulation = settings["Simulation"]["SimulateData"]
    
    if DoSimulation == 1:
        print("No data. The simulation is not done NOW, because it does not provide an image\
              it provides already the particle positions. Thus it is done in \
              nd.get_trajectorie.FindSpots.")
        rawframes_np = 0
        
        
    else:
        data_type        = nd.handle_data.GetVarOfSettings(settings,"File","data_type")
        data_folder_name = nd.handle_data.GetVarOfSettings(settings,"File","data_folder_name")
        data_file_name   = nd.handle_data.GetVarOfSettings(settings,"File","data_file_name")
        use_num_frame   = nd.handle_data.GetVarOfSettings(settings,"File","use_num_frame")
    
        print('start reading in raw images. (That may take a while...)')
        if data_type == 'tif_series':
            if data_folder_name == 0:
                sys.exit('!!! data_folder_name required !!!')
                
            else:
                rawframes_np = nd.handle_data.ReadTiffSeries2Numpy(data_folder_name,use_num_frame)
        
        elif data_type == 'tif_stack':
            if data_file_name == 0:
                sys.exit('!!! data_file_name required !!!')
            else:
                print(data_file_name)
                rawframes_np = nd.handle_data.ReadTiffStack2Numpy(data_file_name)
        
        elif data_type == 'fits':
            if data_file_name == 0:
                sys.exit('!!! data_file_name required !!!')
            else:
                rawframes_np = nd.handle_data.ReadFits2Numpy(data_file_name)   
            
        else:
            sys.exit('Data type %s' %data_type)
        
        print('finishied reading in raw images =)')
        
    return rawframes_np



def ReadTiffStack2Numpy(data_file_name):
    "Reads a tiff stack in"

    rawframes_np = io.imread(data_file_name)
        
    return rawframes_np 



def ReadTiffSeries2Numpy(data_folder_name, use_num_frame):
    """ read a tiff series in """
    
    if use_num_frame == "all":
        use_num_frame = 1000000000
    num_frames_count = 0
    
    rawframes_np = []
#    for fname in os.listdir(data_folder_name):
    for fname in sorted(os.listdir(data_folder_name)): #sorted in case it is unsorted
        print(fname)
        is_tif = fnmatch.fnmatch(fname, '*.tif')
        is_tiff = fnmatch.fnmatch(fname, '*.tiff')
        if is_tif or is_tiff:
            im = Image.open(os.path.join(data_folder_name, fname))
            imarray = np.array(im)
            rawframes_np.append(imarray)
            
            num_frames_count = num_frames_count + 1
            if num_frames_count >= use_num_frame: # break loop if number of frames is reached
                print("Stop reading in after {} frames are read in".format(num_frames_count))
                break
        else:
            print('%s is not a >tif<  file. Skipped it.'%fname)
    
    rawframes_np = np.asarray(rawframes_np) # shape = (60000,28,28)
    
    print('Be sure that tiff series in right order (0002.tif and not 2.tif (which will be sorted after 10.tif))')
    
    
    return rawframes_np



def ReadFits2Numpy(data_file_name):
    """ read a fits image in """
    
    open_fits = pyfits.open(data_file_name)
    rawframes_np = open_fits[0].data
    
    
    return rawframes_np



def UseROI(image, ParameterJsonFile, x_min = None, x_max = None, y_min = None, y_max = None,
           frame_min = None, frame_max = None):
    """ applies a ROI to a given image """ 
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    if settings["ROI"]["Apply"] == 0:
        print("ROI NOT applied")
        image_ROI = image
        
    else:
        print("ROI IS applied")
        x_min = settings["ROI"]['x_min']
        x_max = settings["ROI"]['x_max']
        y_min = settings["ROI"]['y_min']
        y_max = settings["ROI"]['y_max']
        frame_min = settings["ROI"]['frame_min']
        frame_max = settings["ROI"]['frame_max']

        
        image_ROI = image[frame_min : frame_max, y_min : y_max, x_min : x_max]
        
        
        if settings["ROI"]["Save"] == 1:
            data_folder_name = settings["File"]["data_folder_name"]
            SaveROIToTiffStack(image_ROI, data_folder_name)

        
        print("Size rawdata (frames, height, length):", image.shape)
        print("Size ROI (frames, height, length):", image_ROI.shape)
        
    WriteJson(ParameterJsonFile, settings)
    
    return image_ROI



def UseSuperSampling(image_in, ParameterJsonFile, fac_xy = None, fac_frame = None):
    """ supersamples the data in x, y and frames by integer numbers
    
    e.g.: fac_frame = 5 means that only every fifth frame is kept
    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DoSimulation = settings["Simulation"]["SimulateData"]
    
    if DoSimulation == 1:
        print("No data. Do a simulation later on")
        image_super = 0
                
    else:
        # supersampling  
        if settings["Subsampling"]["Apply"] == 0:
            print("Supersampling NOT applied")
            fac_xy = 1
            fac_frame = 1
            
        else:
            print("Supersampling IS applied")
            fac_xy = settings["Subsampling"]['fac_xy']
            fac_frame = settings["Subsampling"]['fac_frame']
            
            
        image_super = image_in[::fac_frame, ::fac_xy, ::fac_xy]
            
            
        settings["MSD"]["effective_fps"] = round(settings["Exp"]["fps"] / fac_frame,2)
        settings["MSD"]["effective_Microns_per_pixel"] = round(settings["Exp"]["Microns_per_pixel"] / fac_xy,5)
        
        
        if (settings["Subsampling"]["Save"] == 1) and (settings["Subsampling"]["Apply"] == 1):
            data_folder_name = settings["File"]["data_folder_name"]
            SaveROIToTiffStack(image_super, data_folder_name)
        
        
        WriteJson(ParameterJsonFile, settings)
    
    return image_super



def SaveROIToTiffStack(image, data_folder_name):
    from skimage import io

    data_folder_name_roi = data_folder_name + "\\ROI"
    if os.path.isdir(data_folder_name_roi) == False:
        os.makedirs(data_folder_name_roi)
        
    data_folder_name_tif = data_folder_name_roi + "\\subimage.tif"


    print("Save ROI and/or supersampled image in: {0}".format(data_folder_name_tif))
    io.imsave(data_folder_name_tif, image)



def RotImages(rawframes_np, ParameterJsonFile, Do_rotation = None, rot_angle = None):
    """ rotate the rawimage by rot_angle """
    import scipy # it's not necessary to import the full library here => to be shortened
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    Do_rotation = settings["PreProcessing"]["Do_or_apply_data_rotation"]
    
    if Do_rotation == True:
        print('Rotation of rawdata: start removing')
        rot_angle = settings["PreProcessing"]["rot_angle"]

    
        if rawframes_np.ndim == 2:
            im_out = scipy.ndimage.interpolation.rotate(rawframes_np, angle = rot_angle, axes=(1, 0), reshape=True, output=None, order=1, mode='constant', cval=0.0, prefilter=True)
        else:
            im_out = scipy.ndimage.interpolation.rotate(rawframes_np, angle = rot_angle, axes=(1, 2), reshape=True, output=None, order=1, mode='constant', cval=0.0, prefilter=True)

        print("Rotation of rawdata: Applied with an angle of %d" %rot_angle)
        
    else:
        im_out = rawframes_np
        print("Rotation of rawdata: Not Applied")

    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    return im_out



def min_rawframes(rawframes_np, display = False):
    """ minimum projection along the frames, display if wanted """
    
    import NanoObjectDetection as nd
    rawframes_min = np.min(rawframes_np,axis=0)
    
    if display == True:
        title = "Background image"
        xlabel = "long. Position [Px]"
        ylabel = "trans. Position [Px]"
        nd.visualize.Plot2DImage(rawframes_min, title, xlabel, ylabel)

    return rawframes_min
    
    

def max_rawframes(rawframes_np, display = False):
    """ maximum projection along the frames, display if wanted """
    rawframes_max = np.max(rawframes_np,axis=0)
    if display == True:
        plt.imshow(rawframes_max)
        
    return rawframes_max
        


def mean_rawframes(rawframes_np, display = False):
    """ calculate the mean along the frames, display if wanted """
    rawframes_mean = np.mean(rawframes_np,axis=0)
    if display == True:
        plt.imshow(rawframes_mean)
        
    return rawframes_mean
      
  
    
def percentile_rawframes(rawframes_np, percentile, display = False):
    """
    Calculated the percentile along the frames
    display if wanted
    """
    rawframes_percentile = np.percentile(rawframes_np, percentile, axis=0)
    if display == True:
        plt.imshow(rawframes_percentile)
        
    return rawframes_percentile
    

    
def are_rawframes_saturated(rawframes_np, ignore_saturation = False):
    """
    Check if rawimages are saturated.
    This is done by looking if the maximum value is 2^x with x an integer which sounds saturated
    e.g. if the max value is 1024, this is suspicious
    """
    brightes_pixel = np.max(rawframes_np)
    
    # is it a multiple of 2^x ... if so it sounds saturated
    hot_pixel = float(np.log2(brightes_pixel + 1)).is_integer()
    if hot_pixel == True:
        if ignore_saturation == False:
            sys.exit("Your Data seems saturated")
        else:
            print("Your Data seems saturated - but you dont care...")
 
    
    
def total_intensity(rawframes_np, display = False):
    """
    tot_intensity: total intensity in each frame
    rel_intensity: relative intensity with respect to the mean
    can be used to remove laser fluctuations
    """
    import NanoObjectDetection as nd
    # intensity in each frame
    tot_intensity = np.sum(rawframes_np,axis=(1,2))
    
    # intensity in each frame relative to the others
    rel_intensity = tot_intensity / np.mean(tot_intensity)

    if display == True:
        nd.visualize.Plot1DPlot(rel_intensity, "Laser Fluctuations", "Frame", "Relative Laser Intensity")
    
        
    return tot_intensity, rel_intensity



def NormImage(image):
    "normalize an image to [0;1]"
    image = image - np.min(image)
    image = image / np.max(image)
    
    return image



def DispWithGamma(image, gamma = 0.5, display = False):
    "gamma correction of an image"
    image = NormImage(image)
    print(gamma)
    image = image ** gamma
    return image
       


def LogData(rawframes):
    """ calculate ln (log_e) of input """
    # Stefan loves the log
    rawframes_log=np.log(rawframes)
    rawframes_log_median=np.median(rawframes_log)
    rawframes_log[rawframes_log==-np.inf]=rawframes_log_median
    rawframes_log[rawframes_log<0]=0     
            
    return rawframes_log
   

#def MaxROI(image):
#    "Return the maximum ROI of a given array"
#    ROI = {'min_x' : 0,
#           'max_x' : image.shape[1]-1,
#           'min_y' : 0,
#           'max_y' : image.shape[2]-1,
#           'min_frame' : 0,
#           'max_frame' : image.shape[0]-1}
#    return ROI
#
#
#def ChangeROI(settings,x_min = False, x_max = False, y_min = False, y_max = False,
#              frame_min = False, frame_max = False):
#    "Change the ROI settings"
#    if x_min != False:
#        settings["ROI"]['x_min'] = x_min
#    if x_max != False:
#        settings["ROI"]['x_max'] = x_max
#    if y_min != False:
#        settings["ROI"]['y_min'] = y_min
#    if y_max != False:
#        settings["ROI"]['y_max'] = y_max
#    if frame_min != False:
#        settings["ROI"]['min_frame'] = frame_min
#    if frame_max != False:
#        settings["ROI"]['max_frame'] = frame_max
#        
#    return settings
#
#
#        
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


