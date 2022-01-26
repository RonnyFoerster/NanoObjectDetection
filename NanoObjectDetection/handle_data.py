# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: Ronny Förster und Stefan Weidlich

This module take care about reading, writing, changing and simple analysis of the rawdata
"""

# Importing neccessary libraries

import shutil
import numpy as np # Library for array-manipulation
import matplotlib.pyplot as plt # Libraries for plotting
import sys
import json
import os 
from PIL import Image
import fnmatch
from skimage import io
from joblib import Parallel, delayed
import multiprocessing


import NanoObjectDetection as nd


def ReadJson(mypath):
    """
    read the json parameter file into a dictionary   
    """
    nd.logger.debug("Read Json file")

    with open(mypath) as json_file:
        settings = json.load(json_file)
    
    # set the logger
    nd.Tools.LoggerSetLevel(settings["Logger"]["level"])
    
    return settings



def WriteJson(mypath, settings):
    """
    write the current settings to a json file
    
    mypath: path to the json file
    settings
    """   
    nd.logger.debug("Write Json file")
    
    with open(mypath, 'w') as outfile:
        json.dump(settings, outfile, indent = 5)



def GetTrajLengthAndParticleNumber(t):
    """ 
    extract ID and trajectory length of the particle with the longest trajectory
    
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
    """
    get the minimum and maximum of an array rounded to the next decade
    
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



def SpecificValueOrSettings(try_value, settings, key, entry):
    """ check if a specific value is given. If not the one out of the settings is used
    
    Arg:
        try_value: if existing, that one is used an written in the settings
    """
    
    
    # if try_value is None that get the value from the settings 
    if try_value != None:
        use_value = try_value
    else:
        if settings == None:
            sys.exit("Either value or settings needed")
        else:
            use_value = settings[key][entry]
    
    settings[key][entry] = use_value
    
    # print it
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



def ReadData2Numpy(ParameterJsonFile, PerformSanityCheck=True):
    """ read the images in
    
    distinguishes between:
    # tif_series
    # tif_stack
    # fits
    """
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    # get the file properties
    data_type = settings["File"]["data_type"]
    data_folder_name = settings["File"]["data_folder_name"]
    data_file_name = settings["File"]["data_file_name"]
    use_num_frame = settings["File"]["use_num_frame"]

    
    nd.logger.warning('start reading in raw images. (That may take a while...)')
    # select the read-in-routine by data type
    if data_type == 'tif_series':
        if data_folder_name == 0:
            nd.logger.error('!!! data_folder_name required !!!')
            sys.exit()
            
        else:
            rawframes_np = nd.handle_data.ReadTiffSeries2Numpy(data_folder_name,use_num_frame)
    
    elif data_type == 'tif_stack':
        if data_file_name == 0:
            nd.logger.error('!!! data_file_name required !!!')
            sys.exit()
        else:
            rawframes_np = nd.handle_data.ReadTiffStack2Numpy(data_file_name)
        
    else:
        sys.exit('Data type %s' %data_type)
    
    nd.logger.info('finishied reading in raw images =)')
        
        
    if PerformSanityCheck == True:
        nd.logger.info("Perform a sanity check for the raw data...")
        # little sanity check
        # check if camera has saved frames doubled
        CheckForRepeatedFrames(rawframes_np)
        
        # check if camera saturation status is given
        max_value = settings["Find"]["SaturatedPixelValue"]
        
        if max_value == "auto":
            rawframes_np, max_value = CheckForSaturation(rawframes_np)
            settings["Find"]["SaturatedPixelValue"] = max_value

    # check bit depth
    # bit_depth, min_value_distance = CalcBitDepth(rawframes_np)


    nd.handle_data.WriteJson(ParameterJsonFile, settings)

    return rawframes_np


def CalcBitDepth(image):
    """
    calculates the bit-depth of the images, which might differ to the bits a pixel has (e.g. 16bit image, but just 12 bit in depth, meaning that value of 0,16,32 occur but not 1-15 or 17-31)
    """
    
    nd.logger.info("Calculate the bit depth of the camera")
    
    # one image is enough for this
    test = image[0,:,:]
    
    # 2d to 1d
    test = test.flatten()
    num_elements = len(test)
    
    wasted_bits = 0
    
    if np.max(test) <= 255:
        nd.logger.info("8 bit image")
        
        bit_depth = 8 # HG was here
        
    else:
        # transfered in 16bit - but does it have a dynamic range of 16bit?    
        finished_loop = False
        
        while finished_loop == False:
        # minimum distance two values are appart
            min_value_distance = 2**(wasted_bits)
            # count elements with digit 0 at the end!
            
            # idea: if bits are waisted - this is visible in the modulo
            # (e.g if only values like 0,8,16,24,32, occur, than the modulo to 2 is always 0)
            num_no_mod = sum(test%min_value_distance == 0)
            
            if num_no_mod == num_elements:
                # bits wasted - try next one
                wasted_bits = wasted_bits + 1
                
            # NOT NEEDED ANYMORE
            # else:
            #     # true bit depth found
            #     finished_loop = True
            #     # test failed
            #     wasted_bits = wasted_bits - 1
        
        bit_depth = 16 - wasted_bits
                
        if wasted_bits == 0:
            nd.logger.info("16bit data - bit depth: {}".format(bit_depth))
        else:
            nd.logger.warning("16bit data - bit depth: {}".format(bit_depth))
    
    min_value_distance = 2**(wasted_bits)        
    
    return bit_depth, min_value_distance
            


def CheckForRepeatedFrames(rawframes_np, diff_frame = [1,2,3,4,5], last_frame = 1000):
    """
    check if images appear several times
    
    Check the pixel-wise difference and check if the maximum occuring difference is 0. Than the images are identical. Do not look only at neighbouring frames, but also in a wider distance (that happend already). 
    
    diff_frames: distance between two analyzed frames
    
    last_frame: last frame that is considered for calculation
    """
    
    # last frame cannot exceed number of existing frames
    num_frames = rawframes_np.shape[0]
    if num_frames  < last_frame:
        last_frame = num_frames 
        
    nd.logger.info("Check first %i frames for repeated frames (camera error)", (last_frame))       
    
    # dont use full frame if it is too large
    if rawframes_np.shape[2] >= 500:
        nd.logger.info("Only use first 500 pixels in x")
        rawframes_np = rawframes_np[:,:,:500]
    
    found_rep_frame = False
    
    for ii in diff_frame:      
        #check if images are saved doubled
        mydiff = rawframes_np[0:last_frame-ii,:,:] - rawframes_np[ii:last_frame,:,:]
        
        #pixel value that differs most
        max_diff_value = np.max(np.abs(mydiff), axis = (1,2))
        
        # number of identical frames
        num_identical_frames = len(max_diff_value[max_diff_value == 0])
        
        if num_identical_frames > 0:
            nd.logger.Warning("%s consecutive images are identical (frame difference is: %s). Probably the camera did something stupid!", num_identical_frames,ii)
            raise ValueError()
        
    if found_rep_frame == False:
          nd.logger.info("... no Camera error detected")       
        

def CheckForSaturation(rawframes_np, warnUser=True):
    """
    check if saturation is present in the raw data
    
    Saturation is visible in the intensity histogramm has a peak in the highest intensity bin.
    """
    nd.logger.info("Check for saturated frames")
    
    # sets the value a pixel has when it is saturated
    max_value = int(np.max(rawframes_np))
    
    # find coordinates of maximum values
    pos = np.where(rawframes_np == max_value)
    
    # frames where a pixel reaches the maximum
    frames = pos[0]
    
    # have every frames only once
    frames = np.unique(frames)
    
    # select first 10 frames, otherwise the following calcuation takes to long
    if len(frames) > 10:
        frames_first_10 = frames[0:10]
    else:
        frames_first_10 = frames
    
    # 8Bit image (works for 10 and 12 bits too)
    num_bins = 2**8
    
    plt.figure()
    plt.hist(np.ndarray.flatten(rawframes_np[frames_first_10,:,:]), bins = num_bins, log = True)
    plt.title("Intensity histogramm of images with bright particles")
    plt.xlabel("Intensity")
    plt.ylabel("Counts")
    
    plt.show(block = False)
    plt.pause(1)
    
    if warnUser==True:
        IsSaturated = nd.handle_data.GetInput("An intensity histogram should be plotted. The highest intensity bin should not be a peak. If you see such a peak, you probably have saturation. But maybe you choose the exposure time to large on purpuse, ignore saturated areas, because your are interested in something very dim. In this case you should treat your data like you have no saturation. Do you have saturation?", ["y", "n"])
        

        if IsSaturated  == "n":
            max_value = 'No Saturation'
        
        if IsSaturated  == "y":
            nd.logger.warning("Saturation suspected. Check your rawimages to find out if they are saturated")
            
            nd.logger.info("Pixel saturate at value: %.0f", max_value)

            # print some statistics
            frames_total = rawframes_np.shape[0]
            frames_sat = len(frames)
            sat_ratio = frames_sat/frames_total*100
            
            nd.logger.warning("Number of frames: Total: %s, Saturated: %s (%s%%) \n", frames_total, frames_sat, sat_ratio)

            
            nd.logger.warning("First 10 frames where saturation occurs: %s", frames_first_10)
            
            SetBackground = nd.handle_data.GetInput("Shall frames with a SINGLE overexposed pixel set to background image (choose >n< if unsure)", ["y", "n"])
            
            if SetBackground == "y":
                rawframes_np[frames,:,:] = np.min(rawframes_np, axis = 0)
            
                nd.logger.warning("Replace a frame where saturation occurs with a background image!")
                    
            else:
                nd.logger.info("Saturated pixels are excluded from evaluation later on!")

                
    return rawframes_np, max_value
    
    

def ReadTiffStack2Numpy(data_file_name):
    """ read a tiff stack in """

    nd.logger.info('read file: %s', data_file_name)

    rawframes_np = io.imread(data_file_name)
        
    return rawframes_np 



def ReadTiffSeries2Numpy(data_folder_name, use_num_frame = "all", ShowProgress = False, CreateSubFolder = False):
    """
    read a tiff series in

    Parameters
    ----------
    data_folder_name : string
        data_folder_name.
    use_num_frame : TYPE, optional
        number of images which are maximal read in. Can have the value "all"
    ShowProgress : TYPE, optional
        DESCRIPTION. The default is False.
    CreateSubFolder : Boolean, optional
        Moves induvidual tif into a subfolder. The default is False.

    Returns
    -------
    image

    """
    
    if ShowProgress == True:
        nd.logger.info('read file: %s', data_folder_name)
    
    if use_num_frame == "all":
        use_num_frame = 1000000000
    num_frames_count = 0
    
    # create the subfolder the images are moved to, if the feature is switched on
    if CreateSubFolder == True:
        path_subfolder = os.path.join(data_folder_name, "tif_series")
        
        #check if directory exists
        if os.path.isdir(path_subfolder) == False:
            if ShowProgress == True:
                nd.logger.info("Create tif subfolder to move images into")
            os.mkdir(path_subfolder) 
            
    
    rawframes_np = []
    
    for fname in sorted(os.listdir(data_folder_name)): #sorted in case it is unsorted
        if ShowProgress == True:
            print("read frame: ", fname)
        else:
            nd.logger.debug("read frame: %s", fname)
        is_tif = fnmatch.fnmatch(fname, '*.tif')
        is_tiff = fnmatch.fnmatch(fname, '*.tiff')
        if is_tif or is_tiff:
            full_path_fname = os.path.join(data_folder_name, fname)
            im = Image.open(full_path_fname)
            imarray = np.array(im)
            rawframes_np.append(imarray)
            
            num_frames_count = num_frames_count + 1
            if num_frames_count >= use_num_frame: # break loop if number of frames is reached
                nd.logger.warning("Stop reading in after %s frames are read in", num_frames_count)
                break
            
            #move image if feature is switched on
            if CreateSubFolder == True:
                full_path_fname_new = os.path.join(path_subfolder, fname)
                nd.logger.debug("store in: %s", full_path_fname_new)
                shutil.move(full_path_fname, full_path_fname_new)


        else:
            nd.logger.debug('%s is not a >tif<  file. Skipped it.', fname)
    
    rawframes_np = np.asarray(rawframes_np)
    
    if ShowProgress == True:
        #Two hints for the use of tiff series
        nd.logger.warning('\n Be sure that tiff series in right order (0002.tif and not 2.tif (which will be sorted after 10.tif))')
        
        nd.logger.info('\n Tiff series need much longer to be read in than a 3D tiff stack, which can be generated out of the tif-series by ImageJ (FIJI) or similar programs.')   
    
    return rawframes_np



def RoiAndSuperSampling(settings, ParameterJsonFile, rawframes_np):
    """
    ROI main function
    """
    
    if settings["Help"]["ROI"] == 1:
        nd.AdjustSettings.FindROI(rawframes_np)

    rawframes_ROI = UseROI(rawframes_np, settings)
        
    return rawframes_ROI



def UseROI(image, settings, x_min = None, x_max = None, y_min = None, y_max = None, frame_min = None, frame_max = None):
    """
    applies a ROI to a given image

    Parameters
    ----------
    image : numpy
        DESCRIPTION.
    settings : TYPE
        DESCRIPTION.
    x_min : TYPE, optional
        DESCRIPTION. The default is None.
    x_max : TYPE, optional
        DESCRIPTION. The default is None.
    y_min : TYPE, optional
        DESCRIPTION. The default is None.
    y_max : TYPE, optional
        DESCRIPTION. The default is None.
    frame_min : TYPE, optional
        DESCRIPTION. The default is None.
    frame_max : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    image_ROI : TYPE
        image out.

     """ 
    
    nd.logger.info("Size rawdata \n (frames, height, width): %s", image.shape)
    
    if settings["ROI"]["Apply"] == 0:
        nd.logger.info("ROI NOT applied")
        image_ROI = image
        
    else:
        nd.logger.info("ROI IS applied")
        x_min = settings["ROI"]['x_min']
        x_max = settings["ROI"]['x_max']
        y_min = settings["ROI"]['y_min']
        y_max = settings["ROI"]['y_max']
        frame_min = settings["ROI"]['frame_min']
        frame_max = settings["ROI"]['frame_max']
       
        image_ROI = image[frame_min : frame_max, y_min : y_max, x_min : x_max]
        
        # save ROI if required
        if settings["ROI"]["Save"] == 1:
            data_folder_name = settings["File"]["data_folder_name"]
            SaveROIToTiffStack(image_ROI, data_folder_name)
                
        nd.logger.info("Size ROI \n (frames, height, width): %s", image_ROI.shape)
    
    return image_ROI



def SaveROIToTiffStack(image, data_folder_name):
    data_folder_name_roi = data_folder_name + "\\ROI"
    if os.path.isdir(data_folder_name_roi) == False:
        os.makedirs(data_folder_name_roi)
        
    data_folder_name_tif = data_folder_name_roi + "\\subimage.tif"


    nd.logger.info("Save ROI and/or supersampled image in: %s", data_folder_name_tif)
    io.imsave(data_folder_name_tif, image)



def min_rawframes(rawframes_np, display = False):
    """ minimum projection along the frames, display if wanted """
    
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
 
    
    
def total_intensity(rawframes_np, display = False):
    """
    Calculated the intensity in each frame. This can be used to remove laser fluctuations or so.

    Parameters
    ----------
    rawframes_np : TYPE
        DESCRIPTION.
    display : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    tot_intensity : TYPE
        total intensity in each frame.
    rel_intensity : TYPE
        relative intensity with respect to the mean. 
    """

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



def DispWithGamma(image, gamma = 0.5):
    "gamma correction of an image"
    image = NormImage(image)
    print(gamma)
    image = image ** gamma
    return image
       


def GetNumberVerbose():
    #decides how many outputs are done in each parallel multiprocessing run
    
    level = nd.logger.getEffectiveLevel()
    
    if level <= 10:
        verbose = 11
    elif level <= 20:
        verbose = 5
    else:
        verbose = 1
        
    return verbose
    
    
def GetInput(InputText, ListAllowedValues):
    """
    This function checks if inserted value of the input function are valid

    Parameters
    ----------
    InputText : STRING
        Text that is displayed on the screen.
    ListAllowedValues : List
        List of allowed values which are accepted as input.

    Returns
    -------
    value : TYPE
        user value that is valid

    """
    
    
    valid = False
    
    while valid == False:
        value = input(str(InputText + " (options: " + str(ListAllowedValues) + "): "))
        
        if value in ListAllowedValues:
            nd.logger.debug("Input accepted")
            valid = True
            
        else:        
            nd.logger.warning("Input declined.Allowed values: %s", ListAllowedValues)
    
    # make int if string is an int
    try:
        value = int(value)
        nd.logger.debug("Input converted into int")
    except:
        nd.logger.debug("Input stays string")
  
    return value



def MakeIntParallel(image_in, dtype):
    """
    Datatype conversion from float to int

    Parameters
    ----------
    image_in : TYPE
        image.
    dtype : STRING
        options: 'int16' or 'uint16'.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    
    # prepare parallelization
    num_cores = multiprocessing.cpu_count()
    num_lines = image_in.shape[1]
    inputs = range(num_lines)
    num_verbose = nd.handle_data.GetNumberVerbose()
    
    # define the function that will be executed later in parallel
    def DoParallel(im_in, dtype):
        if dtype == 'int16':
            im_in = (np.round(im_in)).astype("int16")
        elif dtype == 'uint16':
            im_in = (np.round(im_in)).astype("uint16")
        
        return im_in 
    
    # execute function in parallel for each line
    image_out_list = Parallel(n_jobs=num_cores, verbose = num_verbose)(delayed(DoParallel)(image_in[:,loop_line,:], dtype) for loop_line in inputs)

    # make list to numpy
    image_out = np.asarray(image_out_list)
    
    # make the dimensions right again
    image_out = np.transpose(image_out, (1,0,2))
    
    return image_out



def pandas2csv(my_pandas, save_folder_name, save_file_name, write_index = False):
    """
    save a pandas to a csv

    Parameters
    ----------
    my_pandas : pandas
        variable to save.
    save_folder_name : TYPE
        DESCRIPTION.
    save_file_name : TYPE
        DESCRIPTION.
    write_index : TYPE, optional
        write row names. The default is False.

    Returns
    -------
    None.

    """
    
    # creates full path name and creates folders if required
    my_dir_name, entire_path_file, time_string = nd.visualize.CreateFileAndFolderName(save_folder_name, save_file_name, d_type = 'csv')
    
    nd.logger.info("Save data...")
    
    #export
    my_pandas.to_csv(entire_path_file, index = write_index)
    
    nd.logger.info('Data stored in: %s', format(my_dir_name))
    
    
    
def SaveTifSeriesAsStack_MainDirectory(main_data_folder_name, CreateSubFolder = True, DoParallel = False):
    """
    Loops through a folder with a set of subfolders (different experimental data) which contain tif series and converts each of the series into a single 3d-tif stack (fast for reading in.)


    Parameters
    ----------
    main_data_folder_name : TYPE
        Path to the folder where the required subfolders are in.
    CreateSubFolder : TYPE, optional
        Move tif to subsubdirectory in each subdirectory. The default is True.

    Returns
    -------
    None.

    """      
    
    # get files in given path
    subdir = os.listdir(main_data_folder_name)
    
    
    def LoopFunction(subdir_loop, main_data_folder_name, CreateSubFolder, ShowProgress):
        data_folder_name = main_data_folder_name + "\\" + subdir_loop
        
        # check if file is a directory - continue if TRUE
        if os.path.isdir(data_folder_name) == True:   
            data_tif_name = subdir_loop + ".tif"
            
            # Convert 2d tif list to 3d tif image in subdirectoy
            SaveTifSeriesAsStack(data_folder_name, CreateSubFolder = CreateSubFolder, data_tif_name = data_tif_name, ShowProgress = ShowProgress)
    
    
    if DoParallel == False:
        # loop through all files
        nd.logger.info("Loop through the folders seriell")
        for subdir_loop in subdir:
            print(subdir_loop)
            LoopFunction(subdir_loop, main_data_folder_name, CreateSubFolder, ShowProgress = True)
    
    else:
        nd.logger.info("Loop through the folders parallel")
        # nd.logger.error("THIS IS NOT WORKING CURRENTLY FOR AN UNKNOWN REASON !")
        
        num_cores = multiprocessing.cpu_count()
        num_verbose = nd.handle_data.GetNumberVerbose()
        
        # no more cores than folders
        if num_cores > len(subdir):
            num_cores = len(subdir)
        
        Parallel(n_jobs=num_cores, verbose = num_verbose)(delayed(LoopFunction)(subdir_loop, main_data_folder_name, CreateSubFolder, ShowProgress = False) for subdir_loop in subdir)
        
        
        
    
    
def SaveTifSeriesAsStack(data_folder_name, ShowProgress = True, CreateSubFolder = False, data_tif_name = None):
    """
    Loads all 2d tif images in an directory and stores them as a single 3d tif image/stack    

    Parameters
    ----------
    data_folder_name : TYPE
        path of folder
    ShowProgress : TYPE, optional
        prints current file that is read in. The default is True.
    CreateSubFolder : TYPE, optional
        moves 2d tif in subdirectory. The default is False.
    data_tif_name : TYPE, optional
        customized name of 3d tif stack. The default is None.

    Returns
    -------
    None.
    """
    

    # reads all images
    rawframes_np = ReadTiffSeries2Numpy(data_folder_name, ShowProgress = ShowProgress, CreateSubFolder = CreateSubFolder)
    
    num_frames = rawframes_np.shape[0]
    
    if data_tif_name == None:
        data_folder_name_tif = data_folder_name + "\\3d_stack.tif"
    else:
        data_folder_name_tif = data_folder_name + "\\" + data_tif_name


    if ShowProgress == True:
        print(data_folder_name_tif)

    nd.logger.info("Save 3d tif image ...")
    # saves 3d tif
    io.imsave(data_folder_name_tif, rawframes_np)

    if num_frames > 1000:
        # additional with first 1000 frames
        data_folder_name_tif_1000 = data_folder_name + "\\3d_stack_1000frames.tif"
        
        nd.logger.info("Save first 1000 frames in an extra file for faster testing...")
        io.imsave(data_folder_name_tif_1000, rawframes_np[:1000,:,:])
        
    nd.logger.info("Converting into 3d stack finished.")