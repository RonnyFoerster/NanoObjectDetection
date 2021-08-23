# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:20:48 2019

@author: foersterronny
@small author : Jisoo
# We try new branch develope

set up everything from scratch to evaluate new experimental data
"""

import NanoObjectDetection as nd
import shutil
import os
#from pdb import set_trace as bp #debugger
import json
from tkinter import filedialog



def NewEvaluation():
    # In[Data Path and Type]
    
    datatype = nd.handle_data.GetInput("Is your data stored in : \n 1 - a single file (tif-stack) or \n 2 - multiple files (tif-series) ?", ["1", "2"])

    # select data type
    if datatype == 1:
        data_type = "tif_stack"
    else:
        data_type = "tif_series"
    
    # select the data
    data_file_name = filedialog.askopenfilename(title = "Please select the file", filetypes = (("*tif-files", "*.tif"),("*tiff-files", "*.tiff")))

    nd.logger.info("data_file_name: %s", data_file_name)

    # get data folder
    data_folder_name = os.path.dirname(data_file_name)

    # where shall the evaluation go?
    dir_results = filedialog.askdirectory(title = "Where should the evaluation scripts and results be saved?")
    
    nd.logger.info("store results into: %s", dir_results)
    
    # copy default "auswertung.py" into that folder
    path_aus_origin = os.path.dirname(nd.__file__) + "\\default_auswertung.py"
    path_aus_new = dir_results + "/auswertung.py"
    
    shutil.copy2(path_aus_origin, path_aus_new)
    
    
    # copy default json into that folder
    path_json_origin = os.path.dirname(nd.__file__) + "\\default_json\\default_json.json"
    mypath = dir_results + "/parameter.json"
    
    shutil.copy2(path_json_origin, mypath)

    with open(mypath) as json_file:
        settings = json.load(json_file)
    

    # In[Setup Parameters]
    
    pre_select = nd.handle_data.GetInput("Please select a number: \
          \n The following setups are implemented \
          \n 1 - new \
          \n 2 - Olympus_4x_0-10_plan_n_Olympus_corpus_Basler_AC4096-40um_camera \
          \n 3 - Olympus_10x_0-25_plan_n_Olympus_corpus_Basler_AC4096-40um_camera \
          \n 4 - Zeiss_5x_0-13_HD_Olympus_corpus_Basler_AC4096-40um_camera \
          \n 5 - Zeiss 20x 0_40 epiplan Objective with Basler AC4096-40um_camera\
          \n 6 - Zeiss 10x 0_20 epiplan Objective with Basler AC4096-40um_camera\
          \n 7 - Zeiss 10x 0_3 epiplan neofluar Objective with Basler AC4096-40um_camera\
        \n\n"
        , ["1", "2", "3", "4", "5", "6", "7"])
    
    
    if pre_select in [2,3,4,5,6,7]:
        nd_path = os.path.dirname(nd.__file__)
        
        if pre_select == 2:
            nd.logger.info("Load: Olympus_4x_0-10_plan_n_Olympus_corpus_Basler_AC4096-40um_camera")
            path_json_origin = nd_path + "\\default_json\\setup\\Olympus_4x_0-10_plan_n_Olympus_corpus_Basler_AC4096-40um_camera.json"
            
        elif pre_select == 3:
            nd.logger.info("Load: Olympus_10x_0-25_plan_n_Olympus_corpus_Basler_AC4096-40um_camera")
            path_json_origin = nd_path + "\\default_json\\setup\\Olympus_10x_0-25_plan_n_Olympus_corpus_Basler_AC4096-40um_camera.json"
            
        elif pre_select == 4:
            nd.logger.info("Zeiss_5x_0-13_HD_Olympus_corpus_Basler_AC4096-40um_camera")
            path_json_origin = nd_path + "\\default_json\\setup\\Zeiss_5x_0-13_HD_Olympus_corpus_Basler_AC4096-40um_camera.json"
            
        elif pre_select == 5:
            nd.logger.info("Zeiss 20x 0_40 epiplan Objective with Basler AC4096-40um_camera")
            path_json_origin = nd_path + "\\default_json\\setup\\default_json_20x 0_40 epiplan Objective with Basler cam.json"
            
        elif pre_select == 6:
            nd.logger.info("Zeiss 10x 0_20 epiplan Objective with Basler AC4096-40um_camera")
            path_json_origin = nd_path + "\\default_json\\setup\\default_json_10x 0_20 epiplan Objective with Basler cam.json"
            
        elif pre_select == 7:
            nd.logger.info("Zeiss 10x 0_3 epiplan neofluar Objective with Basler AC4096-40um_camera")
            path_json_origin = nd_path + "\\default_json\\setup\\Zeiss_10x_0-3_plan_neofluar_Olympus_corpus_Basler_AC4096-40um_camera.json"
        
            
            
        with open(path_json_origin) as json_file:
            pre_settings = json.load(json_file)
        
        settings["Exp"]["NA"]                = pre_settings["Exp"]["NA"]
        settings["Exp"]["Microns_per_pixel"] = pre_settings["Exp"]["Microns_per_pixel"]
        settings["Exp"]["gain"]              = pre_settings["Exp"]["gain"] 
        settings["Exp"]["Temperature"]       = pre_settings["Exp"]["Temperature"]
        
    else:
        print("No standard objective given. \n")
        print("Please insert setup parameters: \n")
        settings["Exp"]["NA"]                = float(input("NA = "))
        settings["Exp"]["Microns_per_pixel"] = float(input("Microns per pixel [um/px] = "))
        settings["Exp"]["gain"]              = float(input("gain (if unknown type 0) = "))
        settings["Exp"]["Temperature"]       = float(input("Temperature [K] (22C = 295K) = "))
        if settings["Exp"]["gain"] == 0:
            settings["Exp"]["gain"] = "unknown"
            
            
    # In[Fiber Parameters]
    
    pre_select = nd.handle_data.GetInput("Please select a number: \
          \n The following fibers are implemented \
          \n 1 - new \
          \n 2 - 1250b3 \
          \n 3 - 1276b1_start \
          \n 4 - 18um Lightcage \
          \n\n"
        , ["1", "2", "3", "4"])
    
    
    if pre_select in [2,3]:
        nd_path = os.path.dirname(nd.__file__)
        
        if pre_select == 2:
            nd.logger.info("Load: Fiber 1250b3")
            path_json_origin = nd_path + "\\default_json\\fiber\\1250b3.json"
            nd.logger.warning("RF: FIRST RUN: CHECK IF THAT WORKS PROPERLY!")
            
        elif pre_select == 3:
            nd.logger.info("Load: Fiber 1267b1_start")
            path_json_origin = nd_path + "\\default_json\\fiber\\1267b1_start.json"
            nd.logger.warning("RF: FIRST RUN: CHECK IF THAT WORKS PROPERLY!")
            
        elif pre_select == 3:
            nd.logger.info("Load: LightCage.json")
            path_json_origin = nd_path + "\\default_json\\fiber\\LightCage.json"
            
        with open(path_json_origin) as json_file:
            pre_settings = json.load(json_file)
        
        settings["Fiber"]["Speckle"] = pre_settings["Fiber"]["Speckle"]
        settings["Fiber"]["TubeDiameter_nm"] = pre_settings["Fiber"]["TubeDiameter_nm"]
        settings["Fiber"]["Mode"] = pre_settings["Fiber"]["Mode"]
        settings["Fiber"]["Waist"] = pre_settings["Fiber"]["Waist"]
        
        
    else:
        print("No standard fiber given. \n")
        print("Please insert setup parameters: \n")
        
        settings["Fiber"]["TubeDiameter_nm"] = float(input("Channel Diameter [um] = ")) * 1000
        settings["Fiber"]["Mode"] = str(input("Mode (e.g. gauss): " ))
        settings["Fiber"]["Waist"] = float(input("Beam waist [um] = "))
                  
            
    # In[Experimental Parameters]
    print("Please insert experimental parameters: \n")
    settings["Exp"]["lambda"]            = float(input("lambda [nm] = "))
    settings["Exp"]["fps"]               = float(input("fps = "))
    settings["Exp"]["ExposureTime"]      = float(input("Exposure Time [ms] = ")) / 1000
    
    
    adjustT = nd.handle_data.GetInput("Temperature, solvent and viscosity: \
                                      \n 0 - Default (T = 295.0 K (22°C); solvent = water; viscosity (9.5e-16 Ns/um^2)) \
                                      \n 1 - RECOMMENDED - adjust yourself"
                                      , ["0","1"])
    # adjustT = input("Do you want to use \n[0] the default values for temperature (295.0 K), solvent (water) and viscosity (9.5e-16 Ns/um^2) or \n[1] adjust them?\n")
    if adjustT == 1:
        temp = 273.15 + float(input("Temperature [C] = "))
        settings["Exp"]["Temperature"] = temp # [K]
        solv = input("Solvent (standard is water; dont use quotes): ")
        settings["Exp"]["solvent"] = solv
        settings["Exp"]["Viscosity"] = nd.Experiment.GetViscosity(temp,solv)


    
    settings["File"]["data_file_name"]   = os.path.normpath(data_file_name)
    settings["File"]["data_folder_name"] = os.path.normpath(data_folder_name)
    settings["File"]["data_type"]        = os.path.normpath(data_type)


    # In[Help Functions Parameters]
    help_options = nd.handle_data.GetInput("\nWhich help functions do you want to use? \
            \n 0 - none \
            \n 1 - auto \
            \n 2 - select yourself \n", ["0", "1", "2"])
                
    if help_options == 0:
        nd.logger.info("Switch all help functions off.")
        settings["Help"]["ROI"] = 0
        settings["Help"]["Bead brightness"] = 0
        settings["Help"]["Bead size"] = 0
        settings["Help"]["Separation"] = 0
        settings["Help"]["Drift"] = 0
        
    elif help_options == 1:
        nd.logger.info("Switch recommended help functions on.")
        settings["Help"]["ROI"] = 0
        settings["Help"]["Bead brightness"] = "auto"
        settings["Help"]["Bead size"] = "auto"
        settings["Help"]["Separation"] = "auto"
        settings["Help"]["Drift"] = "auto"

    else:        
        nd.logger.info("Choose the help functions on your own.")
        settings["Help"]["ROI"] = nd.handle_data.GetInput("Do you want help with the >region of intertest (ROI)< ?", ["0", "1"])
       
        
        settings["Help"]["Bead brightness"] = nd.handle_data.GetInput("Do you want help with the >minimal bead brightness< ? \
                \n 0 - no \
                \n manual - manual setting the value with help \
                \n auto - fully automized parameter estimation \n",
                ["0", "manual", "auto"])
            
        
        settings["Help"]["Bead size"] = nd.handle_data.GetInput("Do you want help with the >bead size< ? \
                \n 0 = no \
                \n manual - manual setting the value with help \
                \n auto - fully automized parameter estimation \n",
                ["0", "manual", "auto"])
        
            
        settings["Help"]["Separation"] = nd.handle_data.GetInput("Do you want help with the maximum allowed movement of a particle between two frames >Max Displacement< and the minimal distance to beads must have >Separation Data<? \
                \n 0 = no \
                \n auto - fully automized parameter estimation \n",
                ["0", "auto"])
 
    
        settings["Help"]["Drift"] = nd.handle_data.GetInput("Do you want help how many frames are used to average the drift out of the system >Drift smoothing frames<? \
                \n 0 = no \
                \n auto - fully automized parameter estimation \n",
                ["0", "auto"])


    # In[Save Everything]
    settings["File"]["json"] = mypath.replace("/","\\")

    # save the stuff   
    nd.handle_data.WriteJson(mypath.replace("/","\\"), settings)    

    # print("Make slash and backslash right")

    print("\nGo to \n{}\n in the explorer and open the py-script and json parameter file.".format(dir_results.replace("/","\\")))
    
    
    
