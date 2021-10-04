# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:41:17 2019

@author: Ronny Förster und Stefan Weidlich

This module tries to optimize the required parameters for the reconstruction of the data.

"""

# inport the required packages
import NanoObjectDetection as nd
import numpy as np 
import trackpy as tp
import matplotlib.pyplot as plt



def Main(rawframes_super, rawframes_pre, ParameterJsonFile):
    """
    Runs the various routines for optimiting and estimating the localizing and trackpy parameters of trackpy

    Parameters
    ----------
    rawframes_super : TYPE
        3d rawimage
    rawframes_pre : TYPE
        3d preprocessed rawimage
    ParameterJsonFile : TYPE
        

    Returns
    -------
    None.

    """

    # read in the settings
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    # which help function is on
    mode_separation  = settings["Help"]["Separation"]
    mode_diameter    = settings["Help"]["Bead size"]
    mode_minmass     = settings["Help"]["Bead brightness"]
    mode_intensity   = settings["Help"]["Intensity Jump"]

    # check if input is valid
    if mode_separation not in ["manual", "auto"]:
        nd.logger.error("Need auto or manual in settings[Help][Separation]")   
        
    if mode_diameter not in ["manual", "auto"]:
        nd.logger.error("Need auto or manual in settings[Help][Bead size]")  
        
    if mode_minmass not in ["manual", "auto"]:
        nd.logger.error("Need auto or manual in settings[Help][Bead brightness]")

    if mode_intensity not in ["manual", "auto"]:
        nd.logger.error("Need auto or manual in settings[Help][Intensity Jump]")
    


    # predicts the distance a particle can move between two frames and how close two beads can be without risk of wrong assignment
    if mode_separation == "auto":
        nd.ParameterEstimation.FindMaxDisplacementTrackpy(ParameterJsonFile)        

    # predicts the expected intensity jump
    if mode_intensity == "auto":
        nd.ParameterEstimation.MaxRelIntensityJump(ParameterJsonFile)        
 
    # predicts diameter and minmass, which are connected and must be calculated as a team
    if (mode_diameter == "manual") or (mode_minmass == "manual"):
        # calculate them a bit iterative
        DoDiameter = False
        
        if mode_diameter == "manual":
            # the minmass needs to be guessed first, in order to identifiy particles whose diameter can then be optimized   
            FindMinmass(rawframes_super, rawframes_pre, ParameterJsonFile, DoDiameter = False)
            
        # optimize PSF diameter
        FindDiameter(rawframes_pre, ParameterJsonFile)  

    else:
        #run the full auto routine and optimize both together
        DoDiameter = True
            
    # optimize minmass to identify particle
    FindMinmass(rawframes_super, rawframes_pre, ParameterJsonFile, DoDiameter = DoDiameter)
        
                

def AdjustSettings_Main(rawframes_super, rawframes_pre, ParameterJsonFile):
    """
    This is an old function. Use nd.AdjustSettings.Main instead.
    """
    nd.logger.warning("Function not in use anymore. Use <Main> instead.")
    Main(rawframes_super, rawframes_pre, ParameterJsonFile)


def GetIntegerInput(MessageOnScreen):
    """
    Ask for a valid integer user input

    Parameters
    ----------
    MessageOnScreen : STRING
        Message displayed in the console.

    Returns
    -------
    myinput : INTEGER
        valid user integer.

    """

    # exit variable
    bad_input = True
    
    # run it till a valid input is received
    while bad_input == True:
        
        # get a numerical input here
        myinput = GetNumericalInput(MessageOnScreen)
        print(myinput)
        
        # leave while loop, if received numerical is an integer
        if myinput == int(myinput):
            bad_input = False
        else:
            print("This is not an integer")
            
    return myinput



def GetNumericalInput(MessageOnScreen):
    """
    Ask for a valid integer user input

    Parameters
    ----------
    MessageOnScreen : STRING
        Message displayed in the console.

    Returns
    -------
    myinput : FLOAT
        valid user float.

    """
    
    # exit variable
    bad_input = True
    
    # run it till a valid input is received
    while bad_input == True:
        
        # get the input there
        myinput = input(MessageOnScreen)
        
        # try if the input is a float and exit if so
        try:
            myinput = float(myinput)
            bad_input = False
            
        except ValueError:
            print("This is not a number")
            
    return myinput



def AskDoItAgain():
    """
    ask if a step shall be repeated

    Returns
    -------
    DoItAgain : BOOLEAN
        yes or no (y/n).

    """

    
    answer = nd.handle_data.GetInput("Same problem and optimize value even more?", ["y", "n"])
    
    if answer == 'y':
        DoItAgain = True
    else:
        DoItAgain = False
        
    return DoItAgain



def AskMethodToImprove():
    """
    Subfunction of the semi-automatic particle finding routine. The user IS unsatisfied with the result and is asked WHAT the problem is.

    Returns
    -------
    method : INT
        1 - Bright (isolated) spots not recognized \n
        2 - Spots where nothing is to be seen  \n
        3 - Bright (non-isolated) spots not recognized but you would like to have them both  \n
        4 - Much more recognized spots than I think I have particles
        
    what changes:
        1 - reduce minmass (Intensity threshold) \n
        2 - increase minmass (Intensity threshold) \n
        3 - reduce separation (particle distance) \n
        4 - increase separation (particle distance) \n
    """
    
    # exit variable for while-loop
    valid_answer = False
    
    while valid_answer == False:
        answer = input('What is the problem? \n'
                       '1 - Bright (isolated) spots not recognized \n'
                       '2 - Spots where nothing is to be seen \n'
                       '3 - Bright (non-isolated) spots not recognized but you would like to have them both \n'
                       '4 - Much more recognized spots than I think I have particles \n')
        method = np.int(answer)
        
        #repeat if the input is not valid
        if method in (1,2,3,4) == False:
            nd.logger.warning("Invalid input: choose 1, 2, 3 or 4")
        else:
            valid_answer = True
            
    return method



def AskIfUserSatisfied(QuestionForUser):
    """
    Ask if user is satisfied

    Parameters
    ----------
    QuestionForUser : STRING
        Question to display.

    Returns
    -------
    UserSatisfied : Boolean
        yes (1/ True) or no (0, False).

    """
    
    # exit variable while loop
    valid_answer = False
    
    while valid_answer == False:
        # plot question in console
        answer = input(QuestionForUser + ' (y/n) :')
        
        # check if input is valid
        if answer in ('y','n') == False:
            nd.logger.warning("Invalid input, insert: y or n")
        else:
            valid_answer = True
            
            # set boolean according the users choice
            if answer == 'y':
                UserSatisfied = True
            else:
                UserSatisfied = False
                
    return UserSatisfied
  


def FindMinmass(rawframes_super, rawframes_pre, ParameterJsonFile, DoDiameter = False):
    """
    Estimated the minmass value that trackpy uses in its feature finding routine
    The value have to be choosen such that dim featues are still reconized, while noise is not mistaken as a particle. See also nd.AdjustSettings.AskMethodToImprove

    Parameters
    ----------
    rawframes_super : 3D np.array
        3d rawimage.
    rawframes_pre : 3D np.array
        3d preprocessed image.
    ParameterJsonFile :
    DoDiameter : Boolean, optional
        Defines if various diameters should be tried for optimization. The default is False.

    Returns
    -------
    None.

    """

    # gets the settings
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    # choose Minmass estimation function
    if settings["Help"]["Bead brightness"] in ["manual", 1]:
        # semi-automatic
        FindMinmass_manual(rawframes_pre, ParameterJsonFile)
        
    elif settings["Help"]["Bead brightness"] == "auto":
        # automatic
        settings = nd.ParameterEstimation.MinmassAndDiameterMain(rawframes_super, rawframes_pre, ParameterJsonFile, DoDiameter = DoDiameter)
        
        # save results
        nd.handle_data.WriteJson(ParameterJsonFile, settings)
        
    else:        
        nd.logging.warning("Bead size not adjusted. Use 'manual' or 'auto' if you want to do it.")

    return
    
    
      
    
def FindMinmass_manual(rawframes_pre, ParameterJsonFile, gamma = 0.4):    
    """
    Semi-automatic function to optimize the parameters for particle identification. It runs the bead finding routine and ask the user which problems the result has. According to the problem it tries to improve

    Parameters
    ----------
    rawframes_pre : 3D numpy
        processed 3d raw image
    ParameterJsonFile :
    gamma : float, optional
        Gamma correction for image display. The default is 0.4.

    Returns
    -------
    settings

    """
    
    nd.logger.warning("This routine was changed but never fully testet RF 211001")
    
    # exit variable
    UserSatisfied = False
    
    # defines that the loop is run the first time
    FirstRun = True
    
    while UserSatisfied == False:
        
        # read the settings. To this in the loop, because parameters are changing within it.
        settings = nd.handle_data.ReadJson(ParameterJsonFile)

        # localize particles in first frames
        obj_first = nd.get_trajectorie.FindSpots(rawframes_pre[0:1,:,:], ParameterJsonFile, SaveFig = True, gamma = gamma)
        
        # display full path of image with the labelled identified objects
        nd.logger.info("New image in: {}. Open it!".format(settings["Plot"]["SaveFolder"]))
        
        # only continue if the user is NOT satisfied with the result
        UserSatisfied = AskIfUserSatisfied("Are you satisfied?")
                                           
        if UserSatisfied == True:
            nd.logger.info("User is satisfied with Minmass parameter estimation")
        else:
            # the user is unsatisfied. Now choose which parameter shall be changed and to which value. Either repeat the same routine with a new parameter (DoItAgain) or choose even a new routine.
            
            
            # choose improving routine and parameters
             # check if an optimization was already performed
            if FirstRun == True:
                # first run, change parameters accordingly
                FirstRun = False
                DoItAgain = False       
            else:
                # ask if SAME optimization shall be repeated (with different parameter)
                DoItAgain = AskDoItAgain()
                
            # ask WHICH optimazation routine is wanted
            if DoItAgain == False:
                method = AskMethodToImprove()
        
            
            nd.logger.info("Continue with method: %s", method)
            if method == 1:
                #
                settings["Find"]["tp_minmass"] = GetIntegerInput("Reduce >Minimal bead brightness< from %d to (must be integer): " %settings["Find"]["tp_minmass"])
    
            elif method == 2:
                settings["Find"]["tp_minmass"] = GetIntegerInput("Enhance >Minimal bead brightness< from %d to (must be integer): " %settings["Find"]["tp_minmass"])
    
            elif method == 3:
                settings["Find"]["tp_separation"] = GetNumericalInput("Reduce >Separation data< from %d to (must be integer): " %settings["Find"]["tp_separation"])
                
            else:
                settings["Find"]["tp_separation"] = GetNumericalInput("Enhance >Separation data< from %d to (must be integer): " %settings["Find"]["tp_separation"])
            
            nd.handle_data.WriteJson(ParameterJsonFile, settings)  
    
    
    return settings
    


def FindDiameter(rawframes_pre, ParameterJsonFile):
    """
    Select if Diameter value in trackpy is estiamted manual or automatically

    Parameters
    ----------
    rawframes_pre : 3d np.array
        preprocessed raw image.
    ParameterJsonFile : TYPE


    """
    
    # reads the settings
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    # choose manual option
    if (settings["Help"]["Bead size"] == "manual") or (settings["Help"]["Bead size"] == 1):
        settings["Find"]["tp_diameter"] = FindDiameter_manual(rawframes_pre, settings)
        
    # choose automatic parameter estimation
    elif settings["Help"]["Bead size"] == "auto":
        settings["Find"]["tp_diameter"] = nd.ParameterEstimation.DiameterForTrackpy(settings)
        
    else:
        nd.logger.warning("Bead size not adjusted. Use 'manual' or 'auto' if you want to do it.")
    
    # save returned diameter
    nd.handle_data.WriteJson(ParameterJsonFile, settings)  
    


def FindDiameter_manual(rawframes_pre, settings, AutoIteration = True):
    """
    Optimize the diameter of the Particles \n
    Start with a very low diameter (3px) and run Trackpy on the first 100 frames. 
    A good choosen diameter leads to a even (flat) distributed decimal place of the localized particle position. If the distribution is not flat (user decides), the diameter is increase till its ok.

    Parameters
    ----------
    rawframes_pre : 3d np array
        preprossed 3d input image
    settings : TYPE
        DESCRIPTION.
    AutoIteration : BOOLEAN, optional
        Run the iteration or just plot the current setting. The default is True.

    Returns
    -------
    try_diameter: Best diameter

    """

    # get the parameters for trackpy
    separation = settings["Find"]["tp_separation"]
    minmass    = settings["Find"]["tp_minmass"]
    
    # choose start parameter
    if AutoIteration == True:
        try_diameter = [3.0 ,3.0]
    else:
        try_diameter = settings["Find"]["tp_diameter"]
    
    # if more than 100 frames exist, just choose the first 100 to save time
    if len(rawframes_pre) > 100:
        rawframes_pre = rawframes_pre[0:100,:,:]
    
    # exit variable while-loop
    UserSatisfied = False
    
    while UserSatisfied == False:
        nd.logger.debug('UserSatisfied? : %i', UserSatisfied)
        nd.logger.info("Try diameter: %i", try_diameter[0])

        # localize the particles
        obj_all = tp.batch(rawframes_pre, diameter = try_diameter, minmass = minmass, separation = separation)
              
        # pass following test if no particle is found
        if obj_all.empty == True:
            nd.logger.warning("No Object found.")
            UserSatisfied = False
        else:
            #checks the subpixel accuracy from trackpy. If this is not uniform distributed, the diameter is wrong
            tp.subpx_bias(obj_all)

            # plot it
            plt.draw()
            plt.title("y, spot sizes = {}".format(try_diameter))
            plt.show()
            
            # ask if the user is satisfied, when activated
            if AutoIteration == True:
                plt.pause(3)
                UserSatisfied = AskIfUserSatisfied('The histogramm should be flat. They should not have a dip in the middle! Particles should be detected. \n Are you satisfied?')
                
            else:
                UserSatisfied = True
                print("\n >>> The histogramm should be flat. They should not have a dip in the middle! <<< \n")

            
        if UserSatisfied == False:
            # increase diameter by 2 for next loop
            # TrackPy wants a list, rather a number
            try_diameter = list(np.asarray(try_diameter) + 2)

    
    if AutoIteration == True:
        nd.logger.info('Your diameter should be:', np.asarray(try_diameter))
    
    nd.logger.info("WARNING: IF YOUR BEADSIZE CHANGED YOU MIGHT HAVE TO UPDATE YOUR MINIMAL BRIGHTNESS!")
    
    return try_diameter
 


def FindROI(rawframes_np):
    """
    Show the maximum projection a 3d images to reveal where the ROI is

    Parameters
    ----------
    rawframes_np : 3d np array
        input image.

    Returns
    -------
    None.

    """

    # maximum projection along time/frame axis
    my_max = nd.handle_data.max_rawframes(rawframes_np)
    
    # create plot settings and plot it
    title = "Maximum projection of raw data"
    xlabel = "x [Px]"
    ylabel = "y [Px]"
    nd.visualize.Plot2DImage(my_max, title = title, xlabel = xlabel, ylabel = ylabel)

    nd.logging.info('Choose the ROI of x and y for min and max value accoring your interest. Insert the values in your json file.')





