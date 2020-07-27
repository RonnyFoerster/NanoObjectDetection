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


def GetIntegerInput(MessageOnScreen):
    """
    Ask for an INTEGER input on the console.
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
    """
    Ask for a FLOAT input on the console
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
    """
    Ask if a step shall be repeated
    """
    valid_answer = False
    while valid_answer == False:
        answer = input("Same problem and optimize value even more? (y/n)")
        
        if answer in ('y','n') == False:
            print("Warning: press y or n")
        else:
            valid_answer = True
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
    """
    Ask if user is satisfied
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



def AdjustSettings_Main(rawframes_pre, ParameterJsonFile):
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)

    # optimized the distance a particle can move between two frames and how close to beads can be without risk of wrong assignment
    if settings["Help"]["Separation"] == "auto":
        nd.ParameterEstimation.FindMaxDisplacementTrackpy(ParameterJsonFile)        

    # if beadsize is not auto - the minmass needs to be guess first, in order to identifiy particles whose diameter can than be optimized   
    if settings["Help"]["Bead size"] != "auto":
        nd.AdjustSettings.FindSpot(rawframes_pre, ParameterJsonFile)
        
    # optimize PSF diameter
    nd.AdjustSettings.SpotSize(rawframes_pre, ParameterJsonFile)  
    
    # optimize minmass to identify particle
    num_particles_trackpy = nd.AdjustSettings.FindSpot(rawframes_pre, ParameterJsonFile)

 
    # maybe do this right before the drift correction
    if settings["Help"]["Drift"] == "auto":
        nd.ParameterEstimation.Drift(ParameterJsonFile, num_particles_trackpy)
    
    


def FindSpot(rawframes_pre, ParameterJsonFile):
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    if (settings["Help"]["Bead brightness"] == "manual") or (settings["Help"]["Bead brightness"] == 1):
        obj_first, settings, num_particles_trackpy = FindSpot_manual(rawframes_pre, ParameterJsonFile)
        
    elif settings["Help"]["Bead brightness"] == "auto":
        minmass, num_particles_trackpy = nd.ParameterEstimation.EstimateMinmassMain(rawframes_pre, settings)
        settings["Find"]["Minimal bead brightness"] = minmass
        nd.handle_data.WriteJson(ParameterJsonFile, settings)
        
    else:
        obj_all = nd.get_trajectorie.FindSpots(rawframes_pre[0:1,:,:], ParameterJsonFile)
        num_particles_trackpy = len(obj_all )
        
        print("Bead size not adjusted. Use 'manual' or 'auto' if you want to do it.")

    return num_particles_trackpy
    
    
      
    
def FindSpot_manual(rawframes_pre, ParameterJsonFile, ExternalSlider = False, gamma = 0.4):    
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
                    print("Happy user =)")
                else:
                    # Find out what is wrong
                    method = AskMethodToImprove()
            
            if UserSatisfied == False:              
                print("method:", method)
                if method == 1:
                    settings["Find"]["Minimal bead brightness"] = \
                    GetIntegerInput("Reduce >Minimal bead brightness< from %d to (must be integer): "\
                                      %settings["Find"]["Minimal bead brightness"])
        
                elif method == 2:
                    settings["Find"]["Minimal bead brightness"] = \
                    GetIntegerInput("Enhance >Minimal bead brightness< from %d to (must be integer): "\
                                      %settings["Find"]["Minimal bead brightness"])
        
                elif method == 3:
                    settings["Find"]["Separation data"] = \
                    GetNumericalInput("Reduce >Separation data< from %d to (must be integer): "\
                                      %settings["Find"]["Separation data"])
                    
                else:
                    settings["Find"]["Separation data"] = \
                    GetNumericalInput("Enhance >Separation data< from %d to (must be integer): "\
                                      %settings["Find"]["Separation data"])
                
                nd.handle_data.WriteJson(ParameterJsonFile, settings)  
    
    #get number of located particles by trackpy
    num_particles_trackpy = len(obj_first)
    
    return obj_first, settings, num_particles_trackpy
    


def SpotSize(rawframes_pre, ParameterJsonFile):
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    if (settings["Help"]["Bead size"] == "manual") or (settings["Help"]["Bead size"] == 1):
        settings["Find"]["Estimated particle size"] = SpotSize_manual(rawframes_pre, settings)
        
    elif settings["Help"]["Bead size"] == "auto":
        settings["Find"]["Estimated particle size"] = SpotSize_auto(settings)
        
    else:
        print("Bead size not adjusted. Use 'manual' or 'auto' if you want to do it.")


    
    nd.handle_data.WriteJson(ParameterJsonFile, settings)  
    


def SpotSize_auto(settings):
    
    ImgConvolvedWithPSF = settings["PreProcessing"]["EnhanceSNR"]
    diameter = nd.ParameterEstimation.EstimateDiameterForTrackpy(settings, ImgConvolvedWithPSF)
      
    return diameter



def SpotSize_manual(rawframes_pre, settings, AutoIteration = True):
    """
    Optimize the diameter of the Particles
    """
        
    separation = settings["Find"]["Separation data"]
    minmass    = settings["Find"]["Minimal bead brightness"]

    UserSatisfied = False
    
    if AutoIteration == True:
        try_diameter = [3.0 ,3.0]
    else:
        try_diameter = settings["Find"]["Estimated particle size"]
    
    if len(rawframes_pre) > 100:
        rawframes_pre = rawframes_pre[0:100,:,:]
    
    
    while UserSatisfied == False:
        print('UserSatisfied? : ', UserSatisfied)
        print('Try diameter:' , np.asarray(try_diameter))
#        obj_all = nd.get_trajectorie.batch_np(rawframes_rot, ParameterJsonFile, UseLog = False, diameter = try_diameter)
        obj_all = tp.batch(rawframes_pre, diameter = try_diameter, minmass = minmass, separation = separation)
              

        if obj_all.empty == True:
            UserSatisfied = False
        else:
            tp.subpx_bias(obj_all)

            plt.draw()
            plt.title("y, spot sizes = {}".format(try_diameter))
            plt.show()
            
            if AutoIteration == True:
                plt.pause(3)
                UserSatisfied = AskIfUserSatisfied('The histogramm should be flat. \
                                                   They should not have a dip in the middle! \
                                               Particles should be detected. Are you satisfied?')
                
            else:
                UserSatisfied = True
                print("\n >>> The histogramm should be flat. They should not have a dip in the middle! <<< \n")

            
        if UserSatisfied == False:
            try_diameter = list(np.asarray(try_diameter) + 2)
            

    
#    settings["Find"]["Estimated particle size"] = try_diameter
    
#    nd.handle_data.WriteJson(ParameterJsonFile, settings)  
    
    if AutoIteration == True:
        print('Your diameter should be:', np.asarray(try_diameter))
    
    print("WARNING: IF YOUR BEADSIZE CHANGED YOU MIGHT HAVE TO UPDATE YOUR MINIMAL BRIGHTNESS!")
    
    return try_diameter
 
    

def FindROI(rawframes_np):
    """
    Show the max of all images to show where the ROI is.
    """
    my_max = nd.handle_data.max_rawframes(rawframes_np)
    
    title = "Maximum projection of raw data",
    xlabel = "x [Px]",
    ylabel = "y [Px]"
    nd.visualize.Plot2DImage(nd.handle_data.max_rawframes(rawframes_np),
                             title = title, xlabel = xlabel, ylabel = ylabel)

    print('Chose the ROI of x and y for min and max value accoring your interest. Insert the values in your json file.')





