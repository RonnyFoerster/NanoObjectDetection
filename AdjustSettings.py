# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:41:17 2019

@author: foersterronny
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: foersterronny
"""
#Importing neccessary libraries

#from __future__ import division, unicode_literals, print_function # For compatibility with Python 2 and 3
import numpy as np # Library for array-manipulation
import pandas as pd # Library for DataFrame Handling
import trackpy as tp # trackpy offers all tools needed for the analysis of diffusing particles
import warnings
import sys

import NanoObjectDetection as nd
import matplotlib.pyplot as plt # Libraries for plotting
import matplotlib as mpl # I import this entirely here, as it's needed to change colormaps

from pdb import set_trace as bp #debugger

# In[]
def GetIntegerInput(MessageOnScreen):
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


def FindSpot(rawframes_ROI, ParameterJsonFile):
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    UserSatisfied = False
    FirstRun = True
    
    while UserSatisfied == False:
        obj_first, settings = nd.get_trajectorie.OptimizeParamFindSpots(rawframes_ROI, ParameterJsonFile, SaveFig = True, gamma = 0.7)
        
        plt.pause(3)
    
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
    
    return obj_first
    


def SpotSize(rawframes_rot, ParameterJsonFile):
    UserSatisfied = False
    try_diameter = (3,3)
    while UserSatisfied == False:
        print('UserSatisfied? : ', UserSatisfied)
        print('Try diameter:' , np.asarray(try_diameter))
        obj_all = nd.get_trajectorie.batch_np(rawframes_rot, ParameterJsonFile, UseLog = False, diameter = try_diameter)
        tp.subpx_bias(obj_all)
        plt.draw()
        plt.show()
        plt.pause(3)
        UserSatisfied = AskIfUserSatisfied('The histogramm should be flat. They should not have a dip in the middle!. Are you satisfied?')
        
        if UserSatisfied == False:
            try_diameter = list(np.asarray(try_diameter) + 2)
            
    
    print('Your diameter should be (update JSON manually):', np.asarray(try_diameter))
    
    return
    

def FindROI(rawframes_np):
    my_max = nd.handle_data.max_rawframes(rawframes_np)
    plt.imshow(my_max)
    print('Chose the ROI of x and y for min and max value accoring your interest. Insert the values in your json file.')

