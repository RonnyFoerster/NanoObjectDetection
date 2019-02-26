# NanoObjectDetection


# JSON settings file

{
   "Exp": {
      "NA": 0.13,
      "lambda_nm": 532,
      "fps": 97.281,
      "ExposureTime_s": 0.01028,
      "Microns_per_pixel": 0.63,
      "Temperature": 295.0,
      "const_Boltz": 1.38e-23,
      "data_file_extension": "tif",
      "data_file_name": "C:\\Data\\Mona_Nissen\\40nmgold.fits",
      "data_folder_name": "C:\\Data\\Mona_Nissen",
      "data_type": "fits",
      "DefaultParameterJsonFile": "Z:\\Datenauswertung\\19_ARHCF\\190802\\190225_default.json"
   },
   "ROI": {
      "UseROI": 1,
      "x_min": 0,
      "x_max": 100,
      "y_min": 1,
      "y_max": 101,
      "frame_min": 2,
      "frame_max": 102
   },
   "PreProcessing": {
      "Remove_CameraOffset": 1,
      "Remove_Laserfluctuation": 1,
      "Remove_StaticBackground": 1,
      "RollingPercentilFilter": 0,
      "RollingPercentilFilter_rolling_step": 10,
      "RollingPercentilFilter_rolling_length": 200,
      "RollingPercentilFilter_percentile_filter": 80,
      "ClipNegativeValue": 0,
      "Do_or_apply_data_rotation": 0,
      "rot_angle": 0.3
   },
   "Help": {
      "Help": 1
   },
   "Find": {
      "Analyze in log": 0,
      "Estimated particle size (log-scale) [px]": [
         7.0,
         7.0
      ],
      "Estimated particle size [px]": [
         9.0,
         9.0
      ],
      "Minimal bead brightness": 15,
      "Separation data": 1.5 
   },
   "Link": {
      "Dark time [frames]": 5,
      "Dwell time stationary objects": 100,
      "Max displacement [px]": 5,
      "Max displacement fix [px]": 0.3,
      "Min tracking frames before drift": 15.0,
      "Min_tracking_frames": 65.0
   },
   "StationaryObjects": {
      "Analyze fixed spots": 1,
      "Min distance to stationary object [px]": 20
   },
   "Split": {
      "Max rel median intensity step": 0.5      
   },
   "Drift": {
      "Do transversal drift correction": 1.0,
      "Drift smoothing frames": 20,
      "Drift rolling window size": 5,
      "Min particle per block": 40.0
   },
   "Old": {        
      "Min displacement [px]": 5,
      "Rolling window size": 5      
   },
   "MSD":{
      "Amount summands": 30,
      "Amount lagtimes auto": 1,  
      "Amount lagtimes": 5
   },
   "Plot": {
      "Background_Show": 1,
      "Background_Save": 1,
      "Laserfluctuation_Show": 1,
      "Laserfluctuation_Save": 1,
      "MSD_fit_Show": 1,
      "MSD_fit_Save": 1,
      "Histogramm_Show": 1,
      "Histogramm_Save": 1,
      "Bins": 50,
      "Cutoff Size [nm]": 200,
      "fontsize": 14,
      "SaveFolder": "Z:\\Datenauswertung\\19_ARHCF\\",
      "SaveProperties": "Z:\\Datenauswertung\\19_ARHCF\\PlotProperties.json",
      "dpi": 120
   }
   
}
