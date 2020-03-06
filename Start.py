# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:20:48 2019

@author: foersterronny

Setup everything from scratch to evaluate new experimental data
"""

from pdb import set_trace as bp #debugger
import NanoObjectDetection as nd
import shutil
import os
from pdb import set_trace as bp #debugger
import json
from tkinter import filedialog


# In[]
def NewEvaluation():
    #where is the data
    good_answer = False
    while good_answer == False:
        datatype = input("Is your data stored in : 1 - a single file (tif-stack) or 2 - multiple files (tif-series) (1 or 2)? ")
        if datatype in ["1","2"]:
            # select the data
            data_file_name = filedialog.askopenfilename(title = "Please select the file", filetypes = (("*tiff-files", "*.tiff"),("*tif-files", "*.tif")))
            good_answer = True 
            
            # select data type
            if datatype == "1":
                data_type = "tif_stack"
            else:
                data_type = "tif_series"
                        
        else:
            print("Please press 1 or 2.")

    # get data folder
    data_folder_name = os.path.dirname(data_file_name)

    #where should the evaluation go
    dir_results = filedialog.askdirectory(title = "Where should the evaluation scripts and results be saved?")
    
    #copy default "auswertung.py" into that folder
    path_aus_origin = os.path.dirname(nd.__file__) + "\\default_auswertung.py"
    path_aus_new = dir_results + "/auswertung.py"
    
    shutil.copy2(path_aus_origin, path_aus_new)
    
    
    #copy default json into that folder
    path_json_origin = os.path.dirname(nd.__file__) + "\\default_json.json"
    mypath = dir_results + "/parameter.json"
    
    shutil.copy2(path_json_origin, mypath)



    #update default json with the knowledge you have
    with open(mypath) as json_file:
        settings = json.load(json_file)
    
    pre_select = int(input("Please select a number: \n The following setups are implemented \
          \n 1 - new \
          \n 2 - 5x Objective on Zeiss Micrscope with Basler cam \
          \n\n"))
    
    if pre_select == 2:
        print("load: 5x Objective on Zeiss Micrscope with Basler cam")
        path_json_origin = os.path.dirname(nd.__file__) + "\\default_json_5x_zeiss_cam_basler.json"
        
        with open(path_json_origin) as json_file:
            pre_settings = json.load(json_file)
        
        settings["Exp"]["NA"]                = pre_settings["Exp"]["NA"]
        settings["Exp"]["Microns_per_pixel"] = pre_settings["Exp"]["Microns_per_pixel"]
        settings["Exp"]["gain"]              = pre_settings["Exp"]["gain"] 
        settings["Exp"]["Temperature"]       = pre_settings["Exp"]["Temperature"]
        
        
    else:
        print("please insert setup parameters: \n")
        settings["Exp"]["NA"]                = float(input("NA = "))
        settings["Exp"]["Microns_per_pixel"] = float(input("Microns per pixel [um/px] = "))
        settings["Exp"]["gain"]              = float(input("gain (if unknown type 0) = "))
        settings["Exp"]["Temperature"]       = float(input("Temperature [K] (22C = 295K) = "))
        if settings["Exp"]["gain"] == 0:
            settings["Exp"]["gain"]           = "unknown"
     
    print("please insert experimental parameters: \n")
    settings["Exp"]["lambda"]            = float(input("lambda [nm] = "))
    settings["Exp"]["fps"]               = float(input("fps = "))
    settings["Exp"]["ExposureTime"]      = float(input("Exposure Time [ms] = ")) / 1000
    
    

    
    
    print("viscocity not inserted yet")
#    settings["Exp"]["Viscocity"] = float(input(: 9.5e-16,
#    settings["Exp"]["Viscocity_auto"] = float(input(: 0,
#    settings["Exp"]["solvent"] = input(: "water",


    settings["Fiber"]["TubeDiameter_nm"] = float(input("Channel Diameter [um] = ")) * 1000
    
    settings["File"]["data_file_name"]   = os.path.normpath(data_file_name)
    settings["File"]["data_folder_name"] = os.path.normpath(data_folder_name)
    settings["File"]["data_type"]        = os.path.normpath(data_type)

    print("Here come the help functions:")
    
    help_options = int(input("Which help functions do you want to use? \
            \n 0 - none \
            \n 1 - auto \
            \n 2 - select myself \n"))
    
    if help_options in [0,1,2]:
        print("invalid input. Apply auto.")
        help_options = 1
    
    if help_options == 0:
        print("switch all help functions off.")
        settings["Help"]["ROI"] = 0
        settings["Help"]["Bead brightness"] = 0
        settings["Help"]["Bead size"] = 0
        settings["Help"]["Separation"] = 0
        settings["Help"]["Drift"] = 0
        
    elif help_options == 1:
        print("switch recommended help functions on.")
        settings["Help"]["ROI"] = 0
        settings["Help"]["Bead brightness"] = "auto"
        settings["Help"]["Bead size"] = "auto"
        settings["Help"]["Separation"] = "auto"
        settings["Help"]["Drift"] = "auto"

    else:        
        print("Choose the help functions on your own.")
        settings["Help"]["ROI"]              = int(input("Do you want help with the >region of intertest (ROI)< ? \
                \n 0 - no \
                \n 1 - yes \n"))
        
        settings["Help"]["Bead brightness"]  = str(input("Do you want help with the >minimal bead brightness< ? \
                \n 0 - no \
                \n manuel - manuel setting the value with help \
                \n auto - fully automized parameter estimation \n"))
        
        settings["Help"]["Bead size"]        = str(input("Do you want help with the >bead size< ? \
                \n 0 = no \
                \n manuel - manuel setting the value with help \
                \n auto - fully automized parameter estimation \n"))
        
        settings["Help"]["Separation"]        = str(input("Do you want help with the maximum allowed movement of a particle between two frames >Max Displacement< and the minimal distance to beads must have >Separation Data<? \
                \n 0 = no \
                \n auto - fully automized parameter estimation \n"))
 
        settings["Help"]["Drift"]        = str(input("Do you want help how many frames are used to average the drift out of the system >Drift smoothing frames<? \
                \n 0 = no \
                \n auto - fully automized parameter estimation \n"))


    settings["File"]["json"] = mypath.replace("/","\\")

    # save the stuff   
    nd.handle_data.WriteJson(mypath.replace("/","\\"), settings)    


    print("Go to {} in the explorer and open the py-script and json parameter file.".format(dir_results))
    
