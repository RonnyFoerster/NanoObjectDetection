# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:02:50 2019

@author: foersterronny
"""
import sys
import platform
import trackpy as tp
import pandas as pd
#from pdb import set_trace as bp #debugger
from distutils.spawn import find_executable
import NanoObjectDetection as nd
import shutil
import os
from packaging import version



def CheckAll(ParameterJsonFile):
    """ main function
    """
    CheckPython()
    CheckTrackpy()
    CheckPanda()
    # CheckLatex()
    CheckJson_Exist(ParameterJsonFile)
    CheckJson_Entires(ParameterJsonFile)
    
def CheckPython():
    """ check if the python version is right
    """
#    python_minimum_versions = '3.6.5'
    python_minimum_versions = '3.6.4'
    python_version = platform.python_version()
    
    if version.parse(python_version) >= version.parse(python_minimum_versions):
        print("Python version valid: ", python_version)
    else:
        print("Python minimum version: ", python_minimum_versions)
        print("Your python version: ", python_version)
        sys.exit("Change your python version accordingly, or insert your python version in python_allowed_versions")
    

def CheckTrackpy():
    """ check if the trackpy version is right
    """
    
    tp_minimum_versions = '0.4'
    tp_version = tp.__version__
    
    if version.parse(tp_version) >= version.parse(tp_minimum_versions):
        print("Trackpy version valid: ", tp_version)
    else:
        print("Trackpy minimum versions: ", tp_minimum_versions)
        print("Your trackpy versions: ", tp_version)
        sys.exit("Change your trackpy version accoringly, or insert your trackpy version in tp_allowed_versions")
       
    CheckNumbda()
         
        
def CheckPanda():
    """ check if the pandas version is right
    """
    
    pd_maximum_versions = '0.23.4'
    pd_version = pd.__version__
    
    if version.parse(pd_version) <= version.parse(pd_maximum_versions):
        print("Pandas version valid: ", pd_version)
    else:
        print("Pandas maximum versions: ", pd_maximum_versions)
        print("Your pandas versions: ", pd_version)
        print("New panda versions do not work since https://github.com/soft-matter/trackpy/issues/529#issue-410397797")
        print("Try: Downgrading your system in Anaconda promt using >>> conda install pandas=0.23.4 <<<")
        sys.exit("Change your pandas version accoringly, or insert your pandas version in pd_maximum_versions")       
    

def CheckLatex():
    # https://stackoverflow.com/questions/40894859/how-do-i-check-from-within-python-whether-latex-and-tex-live-are-installed-on-a
    if find_executable('latex'):
        print("Latex installed")    
    else:
        sys.exit("Latex not installed for making good figures")
        

def CheckJson_Exist(ParameterJsonFile):
    """ check if json file exists, otherwise create a default one
    """
    try:
        settings = nd.handle_data.ReadJson(ParameterJsonFile)

    except ValueError:
        print("JSON file corrupted. Maybe a missing or additional >>>,<<< ?")
     
    except FileNotFoundError:
        print("\n No Json File found in >> {} <<".format(ParameterJsonFile))

        CopyJson = input("Copy standard json (y/n)? ")
        if CopyJson == 'y':
            print("Try copying standard json file")
            nd_path = os.path.dirname(nd.__file__)
            json_name = "default_json.json"
            source_path_default_json = nd_path + "\\" + json_name
            copy_to_path = os.path.dirname(ParameterJsonFile)
            
            shutil.copy2(source_path_default_json, copy_to_path)
            
            previous_name = copy_to_path + "\\" + json_name
            os.rename(previous_name, ParameterJsonFile)
            
            # write JsonPath into Json itself
            settings = nd.handle_data.ReadJson(ParameterJsonFile, CreateNew = True)
            
            print("Done")
        else:
            print("Abort")



def CheckJson_Entires(ParameterJsonFile):
    # check if all required entire exist, otherwise copy from json
    
    # memory if sth was wrong
    MissingEntry = False
    
    # get path of standard json
    nd_path = os.path.dirname(nd.__file__)
    source_path_default_json = nd_path + "\\default_json\\default_json.json"
    
    # read default settings
    settings_default = nd.handle_data.ReadJson(source_path_default_json, CreateNew=True)
    
    # read experimental settings
    settings = nd.handle_data.ReadJson(source_path_default_json, CreateNew=True)
    
    # loop through both levels of keys and check if all required keys exist
    list_key_lv1 = settings_default.keys() # level 1 keys
        
    for loop_key_lv1 in list_key_lv1:
        list_key_lv2 = settings_default[loop_key_lv1].keys() #get level 2 keys
        
        for loop_key_lv2 in list_key_lv2:
            
            # test if key exists
            if (loop_key_lv2 in settings[loop_key_lv1].keys()) == False:
                print("Parameter settings['%s']['%s'] not found" %(loop_key_lv1, loop_key_lv2))
                
                # copy defaul value 
                default_value = settings_default[loop_key_lv1][loop_key_lv2]
                settings[loop_key_lv1][loop_key_lv2] = default_value
                print("Copy default value: ", default_value)
            
                MissingEntry = True
        
    if MissingEntry == True:
        print("Some entries have been missing in json parameter file.")    
    else:
        print("All required entries inside json parameter file.")    
    
    
    
def CheckNumbda():
    # http://soft-matter.github.io/trackpy/v0.4.2/tutorial/performance.html
    
    from trackpy.diag import performance_report
    
    print("Performance report: Is numba installed and working?")
    performance_report()
    