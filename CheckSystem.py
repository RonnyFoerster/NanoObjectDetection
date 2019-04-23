# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:02:50 2019

@author: foersterronny
"""
import sys
import platform
import trackpy as tp
import pandas as pd
from pdb import set_trace as bp #debugger
from distutils.spawn import find_executable

# In[] check python version

def CheckAll():
    """
    Main Function
    """
    CheckPython()
    CheckTrackpy()
    CheckPanda()
    CheckLatex()
    
def CheckPython():
    """
    Checks if the python version is right
    """
    python_minimum_versions = '3.6.5'
    python_version = platform.python_version()
    
    if python_version >= python_minimum_versions:
        print("Python version valid: ", python_version)
    else:
        print("Python minimum versions: ", python_minimum_versions)
        print("Your python versions: ", python_version)
        sys.exit("Change your python version accoringly, or insert your python version in python_allowed_versions")
    

# In[] check trackpy version    
def CheckTrackpy():
    """
    Checks if the trackpy version is right
    """
    
    tp_minimum_versions = '0.4'
    tp_version = tp.__version__
    
    if tp_version >= tp_minimum_versions:
        print("Trackpy version valid: ", tp_version)
    else:
        print("Trackpy minimum versions: ", tp_minimum_versions)
        print("Your trackpy versions: ", tp_version)
        sys.exit("Change your trackpy version accoringly, or insert your trackpy version in tp_allowed_versions")
        
        
        
    
    # In[] check trackpy version    
def CheckPanda():
    """
    Checks if the panda version is right
    """
    
    pd_maximum_versions = '0.23.4'
    pd_version = pd.__version__
    
    if pd_version <= pd_maximum_versions:
        print("Pandas version valid: ", pd_version)
    else:
        print("Pandas maximum versions: ", pd_maximum_versions)
        print("Your trackpy versions: ", pd_version)
        print("New panda versions do not work since https://github.com/soft-matter/trackpy/issues/529#issue-410397797")
        print("Try: Downgrading your system in Anaconda promt using >>> conda install pandas=0.23.4 <<<")
        sys.exit("Change your pandas version accoringly, or insert your pandas version in pd_maximum_versions")
        
    

def CheckLatex():
#    https://stackoverflow.com/questions/40894859/how-do-i-check-from-within-python-whether-latex-and-tex-live-are-installed-on-a
    if find_executable('latex'):
        print("Latex installed")    
    else:
        sys.exit("Latex not installed for making good figures")
        
