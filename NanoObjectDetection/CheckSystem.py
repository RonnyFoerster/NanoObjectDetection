# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:02:50 2019

This module checks if all packages have the correct version and that the json parameter file is in a proper state

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
    
    nd.logger.info("Check Packages of Python: starting...")
    
    CheckPython()
    CheckTrackpy()
    CheckPanda()
    
    nd.logger.info("Check inserted json parameter file: starting...")
    settings = CheckJson_Exist(ParameterJsonFile)
    settings = CheckJson_path(ParameterJsonFile, settings, CreateNew = False)
    settings = CheckJson_specify_default_auto(settings)
    settings = CheckJson_Entries(settings)
    
    nd.logger.info("Check Packages and json: ...finished")
    
    nd.handle_data.WriteJson(ParameterJsonFile, settings)
    
    SetupLogger(settings, TryLogger = True)
 

def LogWhereEnvIs():
    nd.logger.info("Have a look into the 'PythonEnvironments' Folder in the nd-package.")
 
    
def CheckPython():
    """ check if the python version is right
    """
#    python_minimum_versions = '3.6.5'
    python_minimum_versions = '3.6.4'
    python_version = platform.python_version()
    
    if version.parse(python_version) >= version.parse(python_minimum_versions):
        nd.logger.info("Python version valid: %s", python_version)
    else:
        nd.logger.critical("Python minimum version: %s", python_minimum_versions)
        nd.logger.critical("Your python version: %s", python_version)
        LogWhereEnvIs()
        sys.exit("Change your python version accordingly, or insert your python version in python_allowed_versions")
    

def CheckTrackpy():
    """ check if the trackpy version is right
    """
    
    tp_minimum_versions = '0.4'
    tp_version = tp.__version__
    
    if version.parse(tp_version) >= version.parse(tp_minimum_versions):
        nd.logger.info("Trackpy version valid: %s", tp_version)
    else:
        nd.logger.critical("Trackpy minimum versions: %s", tp_minimum_versions)
        nd.logger.critical("Your trackpy versions: %s", tp_version)
        LogWhereEnvIs()
        sys.exit("Change your trackpy version accoringly, or insert your trackpy version in tp_allowed_versions")
       
    # checks the internal fast processing routines
    CheckNumbda()
         
        
def CheckPanda():
    """ check if the pandas version is right
    """
    
    pd_maximum_versions = '0.23.4'
    pd_version = pd.__version__
    
    if version.parse(pd_version) >= version.parse(pd_maximum_versions):
        nd.logger.info("Pandas version valid: %s", pd_version)
    else:
        nd.logger.critical("Pandas maximum versions: %s", pd_maximum_versions)
        nd.logger.critical("Your pandas versions: %s", pd_version)
        nd.logger.warning("New panda versions do not work since https://github.com/soft-matter/trackpy/issues/529#issue-410397797")
        nd.logger.warning("Try: Downgrading your system in Anaconda promt using >>> conda install pandas=0.23.4 <<<")
        LogWhereEnvIs()
        sys.exit("Change your pandas version accoringly, or insert your pandas version in pd_maximum_versions")       



def CheckNumbda():
    # checks the internal fast processing routines
    # http://soft-matter.github.io/trackpy/v0.4.2/tutorial/performance.html
    
    #from tp.diag.performance_report()
    numba_works = tp.try_numba.NUMBA_AVAILABLE

    
    if numba_works:
        nd.logger.info("FAST: numba is available and enabled (fast subnets and feature-finding).")
    else:
        nd.logger.warning("SLOW: numba was not found")
    
        

def CheckJson_Exist(ParameterJsonFile):
    """ check if json file exists, otherwise create a default one
    """
    try:
        settings = nd.handle_data.ReadJson(ParameterJsonFile)

    except ValueError:
        nd.logger.critical("JSON file corrupted!!! Maybe a missing or additional >>>,<<< ? Mabe replace all \\ by \\\\")
     
    except FileNotFoundError:
        nd.logger.warning("No Json File found in \n %s ", ParameterJsonFile)

        CopyJson = nd.handle_data.GetInput("Copy standard json?", ["y", "n"])
        
        if CopyJson == 'y':
            nd.logger.info("Try copying standard json file")
            
            #get standard path name of nd module directory
            nd_path = os.path.dirname(nd.__file__)
            json_name = "default_json.json"
            
            # copy standard json to the path where none was found
            source_path_default_json = nd_path + "\\" + json_name
            copy_to_path = os.path.dirname(ParameterJsonFile)
            shutil.copy2(source_path_default_json, copy_to_path)
            
            # rename defaul json file to customized name
            previous_name = copy_to_path + "\\" + json_name
            os.rename(previous_name, ParameterJsonFile)
            
            # write JsonPath into Json itself
            settings = nd.handle_data.ReadJson(ParameterJsonFile, CreateNew = True)
            
            nd.logger.info("Done")
        else:
            nd.logger.error("Abort")

    return settings


def CheckJson_Entries(settings):
    """
    check if all required entires exist, otherwise copy from json \n
    update old keys
    """
    
    #check if old keys have the new name - rename otherwise
    if "Estimated particle size" in settings["Find"].keys():
        settings["Find"]["tp_diameter"] = settings["Find"]["Estimated particle size"]
        del settings["Find"]["Estimated particle size"]
        nd.logger.warning("Replace key <Estimated particle size> by <tp_diameter>")
        
    if "Minimal bead brightness" in settings["Find"].keys():
        settings["Find"]["tp_minmass"] = settings["Find"]["Minimal bead brightness"]
        del settings["Find"]["Minimal bead brightness"]
        nd.logger.warning("Replace key <Minimal bead brightness> by <tp_minmass>")
        
    if "Separation data" in settings["Find"].keys():
        settings["Find"]["tp_separation"] = settings["Find"]["Separation data"]
        del settings["Find"]["Separation data"]
        nd.logger.warning("Replace key <Separation data> by <tp_separation>")
    
    if "PercentileThreshold" in settings["Find"].keys():
        settings["Find"]["tp_percentile"] = settings["Find"]["PercentileThreshold"]
        del settings["Find"]["PercentileThreshold"]
        nd.logger.warning("Replace key <PercentileThreshold> by <tp_percentile>")   
    
    # memory if sth was wrong
    MissingEntry = False
    
    
    # loop through keys in the default file and check if experimental parameter file needs an update
    # get defaul path
    source_path_default_json = settings["File"]["DefaultParameterJsonFile"]
    
    # read default settings
    settings_default = nd.handle_data.ReadJson(source_path_default_json)
        
    # loop through both levels of keys and check if all required keys exist
    list_key_lv1 = settings_default.keys() # level 1 keys
        
    for loop_key_lv1 in list_key_lv1:
        list_key_lv2 = settings_default[loop_key_lv1].keys() #get level 2 keys
        
        # test if key exists
        if (loop_key_lv1 in settings.keys()) == False:
            nd.logger.info("Parameter settings['%s'] not found" ,(loop_key_lv1))
            settings[loop_key_lv1] = settings_default[loop_key_lv1]
            nd.logger.info("Copy default value.")
            
            MissingEntry = True
        
        for loop_key_lv2 in list_key_lv2:
            
            # test if key exists
            if (loop_key_lv2 in settings[loop_key_lv1].keys()) == False:
                nd.logger.info("Parameter settings['%s']['%s'] not found" ,loop_key_lv1, loop_key_lv2)
                
                # copy defaul value 
                default_value = settings_default[loop_key_lv1][loop_key_lv2]
                settings[loop_key_lv1][loop_key_lv2] = default_value
                nd.logger.info("Copy default value: %s", default_value)
            
                MissingEntry = True
        
    if MissingEntry == True:
        nd.logger.warning("Some entries have been missing in json parameter file. Replaced by the default values")    
    else:
        nd.logger.info("All required entries inside json parameter file.")    
    
    return settings



def CheckJson_path(mypath, settings, CreateNew = False):
    """ check if json path is written in file. otherwise put it there
    
    settings: 
    CreateNew: Sanitiy Check if False
    """    
    
    # check if json path is written in file. otherwise put it there
    if "json" in list(settings["File"].keys()):
        
        # compare given and saved json path
        # lower is needed to convert Lib to lib, which is identical in the explorer
        comp1 = settings["File"]["json"].lower()
        comp2 = mypath.lower()
        if comp1 != comp2:
            if CreateNew == False:
                nd.logger.error("Given Json path does not match defined path in json file!")
                nd.logger.error("You might wanna delete the 'settings' 'json' entry entirely from the json file.")
                nd.logger.error("json path: %s", comp1)
                nd.logger.error("given path: %s", comp2)
                sys.exit()
            else:
                settings["File"]["json"] = mypath

    else:
        settings["File"]["json"] = mypath
    
    return settings

    

def CheckJson_specify_default_auto(settings):
    # set default_json file from default folder
    mypath_default = settings["File"]["DefaultParameterJsonFile"]
    
    # use default_json file if set to default
    if mypath_default == "default":
        mypath_default = os.path.dirname(nd.__file__) + "\\default_json\\default_json.json"
        settings["File"]["DefaultParameterJsonFile"] = mypath_default
    
    try:    
        nd.handle_data.ReadJson(settings["File"]["DefaultParameterJsonFile"])
            
    except:
        nd.logger.error("Default Json file probably not found. You must insert default at the key DefaultParameterJsonFile and run this section AGAIN!")
        sys.exit()
       
    
    # set SaveFolder in case of auto     
    if settings["Plot"]["SaveFolder"] == "auto":
        settings["Plot"]["SaveFolder"] = os.path.dirname(settings["File"]["json"]) + "\\analysis"
        
    # check if saving folders are valid    
    my_path = settings["Plot"]["SaveFolder"]    
    invalid, my_path = CheckIfFolderGeneratable(my_path)
     
    if invalid == True:
        # if path is invalid use the escape path (my_path)
        settings["Plot"]["SaveFolder"] = my_path
    
    nd.logger.info("Figures are saved into: %s", settings["Plot"]["SaveFolder"])
    
    
    
    # same for the save SaveProperties entry
    if settings["Plot"]["SaveProperties"] == "auto":
        settings["Plot"]["SaveProperties"] = settings["Plot"]["SaveFolder"]
    
    # check if saving folders are valid
    my_path = settings["Plot"]["SaveProperties"]
    invalid, my_path = CheckIfFolderGeneratable(my_path)
     
    if invalid == True:
        settings["Plot"]["SaveProperties"] = my_path
    
    nd.logger.info("Properties are saved into: %s", settings["Plot"]["SaveProperties"])
    
  
    return settings



def CheckIfFolderGeneratable(my_path):
    """
    Checks if a folder exists
    """
    
    invalid = False
    # try if path exists
    try:
        os.stat(my_path)
    except:
        # try if the path can be created
        try:
            os.mkdir(my_path)
        except:
            # write on desktop if nothing is working
            my_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
            invalid = True
            nd.logger.warning("Path not accessible. Write on desktop. \n If you are not happy with this, try writing <auto> into the json key!")

    return invalid, my_path
    

    
def SetupLogger(settings, TryLogger = False):    
    #set the logger level
    nd.Tools.LoggerSetLevel(settings["Logger"]["level"], TryLogger = TryLogger)
        

    
    