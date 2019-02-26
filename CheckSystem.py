# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:02:50 2019

@author: foersterronny
"""
import trackpy as tp
import sys
import platform

# In[] check python version

python_allowed_versions = ('3.6.5')
python_version = platform.python_version()

if python_version in python_allowed_versions:
    print("Python version valid: ", python_version)
else:
    print("Allowed python versions: ", tp_allowed_versions)
    print("Your python versions: ", tp_version)
    sys.exit("Change your python version accoringly, or insert your python version in python_allowed_versions")
    

# In[] check trackpy version    
tp_allowed_versions = ('0.4.1', '0.4',)
tp_version = tp.__version__

if tp_version in tp_allowed_versions:
    print("Trackpy version valid: ", tp_version)
else:
    print("Allowed trackpy versions: ", tp_allowed_versions)
    print("Your trackpy versions: ", tp_version)
    sys.exit("Change your trackpy version accoringly, or insert your trackpy version in tp_allowed_versions")
    
    
