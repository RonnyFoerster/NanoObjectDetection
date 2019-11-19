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
    
    print("please insert experimental parameters:")
    
    settings["Exp"]["NA"]                = float(input("NA = "))
    settings["Exp"]["lambda"]            = float(input("lambda [nm] = "))
    settings["Exp"]["fps"]               = float(input("fps = "))
    settings["Exp"]["ExposureTime"]      = float(input("Exposure Time [ms] = ")) / 1000
    settings["Exp"]["Microns_per_pixel"] = float(input("Microns per pixel [um/px] = "))
    settings["Exp"]["gain"]              = float(input("gain (if unknown type 0) = "))
    if settings["Exp"]["gain"] == 0:
       settings["Exp"]["gain"]           = "unknown"
    settings["Exp"]["Temperature"]       = float(input("Temperature [K] (22C = 295K) = "))
    
    print("viscocity not inserted yet")
#    settings["Exp"]["Viscocity"] = float(input(: 9.5e-16,
#    settings["Exp"]["Viscocity_auto"] = float(input(: 0,
#    settings["Exp"]["solvent"] = input(: "water",


    settings["Fiber"]["TubeDiameter_nm"] = float(input("Channel Diameter [um] = ")) * 1000
    
    settings["File"]["data_file_name"]   = os.path.normpath(data_file_name)
    settings["File"]["data_folder_name"] = os.path.normpath(data_folder_name)
    settings["File"]["data_type"]        = os.path.normpath(data_type)

    print("Here come the help functions:")
    settings["Help"]["ROI"]              = int(input("Do you want help with the >region of intertest (ROI)< (0 = no, 1 = yes)?"))
    settings["Help"]["Bead brightness"]  = int(input("Do you want help with the >minimal bead brightness< (0 = no, 1 = yes)?"))
    settings["Help"]["Bead size"]        = int(input("Do you want help with the >bead size< (0 = no, 1 = yes)?"))
    

    settings["File"]["json"] = mypath.replace("/","\\")

    # save the stuff   
    nd.handle_data.WriteJson(mypath.replace("/","\\"), settings)    


    print("Go to {} in the explorer and open the py-script and json parameter file.".format(dir_results))
    
