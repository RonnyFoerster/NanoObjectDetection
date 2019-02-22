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

import nanoobject_detection as nd
import matplotlib.pyplot as plt # Libraries for plotting
import matplotlib as mpl # I import this entirely here, as it's needed to change colormaps



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


def AskIfUserSatisfied():
    valid_answer = False
    while valid_answer == False:
        answer = input('Are you satisfied (y/n)? :')
        if answer in ('y','n') == False:
            print("Warning: press y or n")
        else:
            valid_answer = True
            if answer == 'y':
                UserSatisfied = True
            else:
                UserSatisfied = False
    return UserSatisfied


def FindSpot(rawframes_ROI, settings):
    UserSatisfied = False
    FirstRun = True
    
    while UserSatisfied == False:
        obj_first, settings = nd.get_trajectorie.OptimizeParamFindSpots(rawframes_ROI, settings, SaveFig = True, gamma = 0.4)
    
        if FirstRun == True:
            FirstRun = False
            DoItAgain = False
        else:
            DoItAgain = AskDoItAgain()
            
        if DoItAgain == False:
            # user happy?
            UserSatisfied = AskIfUserSatisfied()
               
            if UserSatisfied == True:
                print("Happy user =)")
                return obj_first, settings
            else:
                # Find out what is wrong
                method = AskMethodToImprove()
                      
        print("method:", method)
        if method == 1:
            settings["Processing"]["Minimal bead brightness"] = \
            GetIntegerInput("Reduce >Minimal bead brightness< from %d to (must be integer): "\
                              %settings["Processing"]["Minimal bead brightness"])

        elif method == 2:
            settings["Processing"]["Minimal bead brightness"] = \
            GetIntegerInput("Enhance >Minimal bead brightness< from %d to (must be integer): "\
                              %settings["Processing"]["Minimal bead brightness"])

        elif method == 3:
            settings["Processing"]["Separation data"] = \
            GetNumericalInput("Reduce >Separation data< from %d to (must be integer): "\
                              %settings["Processing"]["Separation data"])
            
        else:
            settings["Processing"]["Separation data"] = \
            GetNumericalInput("Enhance >Separation data< from %d to (must be integer): "\
                              %settings["Processing"]["Separation data"])
            


        
    
    return settings