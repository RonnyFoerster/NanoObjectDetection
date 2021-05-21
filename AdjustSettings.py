# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:41:17 2019

@author: Ronny Förster und Stefan Weidlich

This module tries to hell the user finding the correct parameters for the analysis
"""


import NanoObjectDetection as nd

import numpy as np # Library for array-manipulation
import trackpy as tp # trackpy offers all tools needed for the analysis of diffusing particles
import matplotlib.pyplot as plt # Libraries for plotting

from pdb import set_trace as bp #debugger



def Main(rawframes_super, rawframes_pre, ParameterJsonFile):
    """
    Runs the various routines for optimiting and estimating the localizing and trackpy parameters of trackpy
    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    mode_separation  = settings["Help"]["Separation"]
    mode_diameter    = settings["Help"]["Bead size"]
    mode_minmass     = settings["Help"]["Bead brightness"]
    mode_intensity   = settings["Help"]["Intensity Jump"]

    #sanity check
    if mode_separation not in ["manual", "auto"]:
        nd.logger.error("Need auto or manual in settings[Help][Separation]")   
        
    if mode_diameter not in ["manual", "auto"]:
        nd.logger.error("Need auto or manual in settings[Help][Bead size]")  
        
    if mode_minmass not in ["manual", "auto"]:
        nd.logger.error("Need auto or manual in settings[Help][Bead brightness]")

    if mode_intensity not in ["manual", "auto"]:
        nd.logger.error("Need auto or manual in settings[Help][Intensity Jump]")
    


    # optimized the distance a particle can move between two frames and how close to beads can be without risk of wrong assignment
    if mode_separation == "auto":
        nd.ParameterEstimation.FindMaxDisplacementTrackpy(ParameterJsonFile)        

    if mode_intensity == "auto":
        nd.ParameterEstimation.MaxRelIntensityJump(ParameterJsonFile)        
 
    # Diameter and minmass are connected and must be calculated as a team
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
    nd.logger.warning("Function not in use anymore. Use <Main> instead.")
    Main(rawframes_super, rawframes_pre, ParameterJsonFile)


def GetIntegerInput(MessageOnScreen):
    """ ask for an INTEGER input on the console
    """
    bad_input = True
    while bad_input == True:
        myinput = GetNumericalInput(MessageOnScreen)
        print(myinput)
        if myinput == int(myinput):
            bad_input = False
        else:
            print("This is not an integer")
    return myinput



def GetNumericalInput(MessageOnScreen):
    """ ask for a FLOAT input on the console
    """
    bad_input = True
    while bad_input == True:
        myinput = input(MessageOnScreen)
        try:
            myinput = float(myinput)
            bad_input = False
        except ValueError:
            print("This is not a number")
    return myinput



def AskDoItAgain():
    """ ask if a step shall be repeated
    """
    
    answer = nd.handle_data.GetInput("Same problem and optimize value even more?", ["y", "n"])
    
    if answer == 'y':
        DoItAgain = True
    else:
        DoItAgain = False
        
    return DoItAgain



def AskMethodToImprove():
    """
    Ask which method shall be applied to improve the particle IDENTIFICATION
    1 - Bright (isolated) spots not recognized \n
    2 - Spots where nothing is to be seen \n
    3 - Bright (non-isolated) spots not recognized but you would like to have them both \n
    4 - Much more recognized spots than I think I have particles \n
    """
    valid_answer = False
    while valid_answer == False:
        answer = input('What is the problem? \n'
                       '1 - Bright (isolated) spots not recognized \n'
                       '2 - Spots where nothing is to be seen \n'
                       '3 - Bright (non-isolated) spots not recognized but you would like to have them both \n'
                       '4 - Much more recognized spots than I think I have particles \n')
        method = np.int(answer)
        
        if method in (1,2,3,4) == False:
            print("Warning: press y or n")
            
        else:
            valid_answer = True
    return method



def AskIfUserSatisfied(QuestionForUser):
    """ ask if user is satisfied
    """
    valid_answer = False
    while valid_answer == False:
        answer = input(QuestionForUser + ' (y/n) :')
        if answer in ('y','n') == False:
            print("Warning: press y or n")
        else:
            valid_answer = True
            if answer == 'y':
                UserSatisfied = True
            else:
                UserSatisfied = False
    return UserSatisfied
  


def FindMinmass(rawframes_super, rawframes_pre, ParameterJsonFile, DoDiameter = False):
    """
    Estimated the minmass value that trackpy uses in its feature finding routine
    The value have to be choosen such that dim featues are still reconized, while noise is not mistaken as a particle
    """    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    if settings["Help"]["Bead brightness"] in ["manual", 1]:
        FindMinmass_manual(rawframes_pre, ParameterJsonFile)
        
    elif settings["Help"]["Bead brightness"] == "auto":
        settings = nd.ParameterEstimation.MinmassAndDiameterMain(rawframes_super, rawframes_pre, ParameterJsonFile, DoDiameter = DoDiameter)
        
        nd.handle_data.WriteJson(ParameterJsonFile, settings)
        
    else:        
        nd.logging.warning("Bead size not adjusted. Use 'manual' or 'auto' if you want to do it.")

    return
    
    
      
    
def FindMinmass_manual(rawframes_pre, ParameterJsonFile, ExternalSlider = False, gamma = 0.4):    
    """
    Main function to optimize the parameters for particle identification
    It runs the bead finding routine and ask the user what problem he has
    According to the problem it tries to improve
    """
    
    UserSatisfied = False
    FirstRun = True
    
    while UserSatisfied == False:
        settings = nd.handle_data.ReadJson(ParameterJsonFile)
        if ExternalSlider == True:
            obj_first = nd.get_trajectorie.FindSpots(rawframes_pre[0:1,:,:], ParameterJsonFile, SaveFig = True, gamma = gamma, ExternalSlider = True)
            UserSatisfied = True
            
        else:
            obj_first = nd.get_trajectorie.FindSpots(rawframes_pre[0:1,:,:], ParameterJsonFile, SaveFig = True, gamma = gamma)
            if FirstRun == True:
                FirstRun = False
                DoItAgain = False
            else:
                DoItAgain = AskDoItAgain()
                
            if DoItAgain == False:
                # user happy?
                my_question = "New image in: {}. Open it! Are you satisfied?".format(settings["Plot"]["SaveFolder"])
                UserSatisfied = AskIfUserSatisfied(my_question)
                   
                if UserSatisfied == True:
                    nd.logger.info("Happy user =)")
                else:
                    # Find out what is wrong
                    method = AskMethodToImprove()
            
            if UserSatisfied == False:              
                nd.logger.info("method: %s", method)
                if method == 1:
                    settings["Find"]["tp_minmass"] = \
                    GetIntegerInput("Reduce >Minimal bead brightness< from %d to (must be integer): "\
                                      %settings["Find"]["tp_minmass"])
        
                elif method == 2:
                    settings["Find"]["tp_minmass"] = \
                    GetIntegerInput("Enhance >Minimal bead brightness< from %d to (must be integer): "\
                                      %settings["Find"]["tp_minmass"])
        
                elif method == 3:
                    settings["Find"]["tp_separation"] = \
                    GetNumericalInput("Reduce >Separation data< from %d to (must be integer): "\
                                      %settings["Find"]["tp_separation"])
                    
                else:
                    settings["Find"]["tp_separation"] = \
                    GetNumericalInput("Enhance >Separation data< from %d to (must be integer): "\
                                      %settings["Find"]["tp_separation"])
                
                nd.handle_data.WriteJson(ParameterJsonFile, settings)  
    

    
    return settings
    


def FindDiameter(rawframes_pre, ParameterJsonFile):
    """
    select if Diameter value in trackpy is estiamted manual or automatically
    """
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    if (settings["Help"]["Bead size"] == "manual") or (settings["Help"]["Bead size"] == 1):
        settings["Find"]["tp_diameter"] = FindDiameter_manual(rawframes_pre, settings)
        
    elif settings["Help"]["Bead size"] == "auto":
        settings["Find"]["tp_diameter"] = nd.ParameterEstimation.DiameterForTrackpy(settings)
        
    else:
        nd.logger.warning("Bead size not adjusted. Use 'manual' or 'auto' if you want to do it.")
    
    nd.handle_data.WriteJson(ParameterJsonFile, settings)  
    


def FindDiameter_manual(rawframes_pre, settings, AutoIteration = True):
    """
    Optimize the diameter of the Particles
    Start with a very low diameter (3px) and run Trackpy on the first 100 frames. 
    A good choosen diameter leads to a even (flat) distributed decimal place of the localized particle position
    If the distribution is not flat, the diameter is increase till its ok.
    """
        
    separation = settings["Find"]["tp_separation"]
    minmass    = settings["Find"]["tp_minmass"]

    UserSatisfied = False
    
    if AutoIteration == True:
        try_diameter = [3.0 ,3.0]
    else:
        try_diameter = settings["Find"]["tp_diameter"]
    
    if len(rawframes_pre) > 100:
        rawframes_pre = rawframes_pre[0:100,:,:]
    
    
    while UserSatisfied == False:
        nd.logger.debug('UserSatisfied? : %i', UserSatisfied)
        nd.logger.info("Try diameter: %i", try_diameter[0])
#        obj_all = nd.get_trajectorie.batch_np(rawframes_rot, ParameterJsonFile, UseLog = False, diameter = try_diameter)
        obj_all = tp.batch(rawframes_pre, diameter = try_diameter, minmass = minmass, separation = separation)
              

        if obj_all.empty == True:
            nd.logger.warning("No Object found.")
            UserSatisfied = False
        else:
            tp.subpx_bias(obj_all)

            plt.draw()
            plt.title("y, spot sizes = {}".format(try_diameter))
            plt.show()
            
            if AutoIteration == True:
                plt.pause(3)
                UserSatisfied = AskIfUserSatisfied('The histogramm should be flat. They should not have a dip in the middle! Particles should be detected. \n Are you satisfied?')
                
            else:
                UserSatisfied = True
                print("\n >>> The histogramm should be flat. They should not have a dip in the middle! <<< \n")

            
        if UserSatisfied == False:
            try_diameter = list(np.asarray(try_diameter) + 2)
            

    
#    settings["Find"]["tp_diameter"] = try_diameter
    
#    nd.handle_data.WriteJson(ParameterJsonFile, settings)  
    
    if AutoIteration == True:
        print('Your diameter should be:', np.asarray(try_diameter))
    
    print("WARNING: IF YOUR BEADSIZE CHANGED YOU MIGHT HAVE TO UPDATE YOUR MINIMAL BRIGHTNESS!")
    
    return try_diameter
 


def FindROI(rawframes_np):
    """ show the maximum value of all images to reveal where the ROI is
    """
    my_max = nd.handle_data.max_rawframes(rawframes_np)
    
    title = "Maximum projection of raw data"
    xlabel = "x [Px]"
    ylabel = "y [Px]"
    nd.visualize.Plot2DImage(my_max, title = title, xlabel = xlabel, ylabel = ylabel)

    nd.logging.info('Choose the ROI of x and y for min and max value accoring your interest. Insert the values in your json file.')





