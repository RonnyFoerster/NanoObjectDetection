# NanoObjectDetection


# JSON settings file
## Important
All the experimental parameters are find in a *.json file. The json datatype is quite strict with the format. The last entry of a key should not end whith a >>> , <<< . Booleans are written as 0 and 1.

## Explenation of parameters

### Exp
Here are all experimental parameters

| Key  | Explanation | Example |
| - | - | - |
| NA  | numberical aperture of the detection objective | 0.13 |
| lambda_nm | wavelength of the detection light (in scattering this is the illumination wavelength) in nm | 532 |
| fps | frames per seconds (fps) of the camera; required for the lagtime (1/fps) |  97.281 |
| ExposureTime_s | exposuretime of each raw image in seconds | 0.01028 |
| Microns_per_pixel | size of a camera pixel in object coordinate. This is given by the pixel pitch of the CCD and the magnification of the system. Be aware that the magnification written on an objective is only valid for the correct tube lens length. | 0.63 |
| Temperature | temperature of the specimen in degree C | 295.0 |
| const_Boltz | Boltzmann constant for the diffusion | 1.38e-23 |
| data_file_extension | File format of the images. | "tif" |
| data_file_name | Complete path of the first image  | "C:\\Data\\Mona_Nissen\\40nmgold.fits" |
| data_folder_name | Location of the images  | "C:\\Data\\Mona_Nissen" |
~~| data_type | File format of the images (old) | "fits" |~~
| DefaultParameterJsonFile | If a json file is incomplete, for example due to an update, the json file is completed by the default value, whose path defined in here. | "Z:\\Datenauswertung\\19_ARHCF\\190802\\190225_default.json" |

### ROI - Region of interest
Here a subregion (ROI) can be defined. This is especially helpfull for debugging, when not the entire image or all frames shall be used for the calculation in order to speed up the calculations. In addition, it can be helpfull if the image is large, but you just wanna evaluate a particle at a very defined region and time.

| Key  | Explanation | Example |
| - | - | - |
| UseROI |  | 1 |
| x_min |  | 0 |
| x_max |  | 100 |
| y_min |  | 25 |
| y_max |  | 80 |
| frame_min |  | 0 |
| frame_max |  | 500 |
  

### PreProcessing
Standard image processing before the specific MSD stuff starts.
   
| Key  | Explanation | Example |
| - | - | - |
| Remove_CameraOffset |  | 1 |
| Remove_Laserfluctuation |  | 1 |
| Remove_StaticBackground |  | 1 |
| RollingPercentilFilter |  | 0 |
| RollingPercentilFilter_rolling_step |  | 10 |
| RollingPercentilFilter_rolling_length |  | 200 |
| RollingPercentilFilter_percentile_filter |  | 80 |
| ClipNegativeValue |  | 0 |
| Do_or_apply_data_rotation |  | 0 |
| rot_angle |  | 0.3 |


### Help
Start a routine that try to help you find the right parameters.

      
| Key  | Explanation | Example |
| - | - | - |
| Help |  | 1 |


### Find
Parameters necessary to run the Tracky module.
        
| Key  | Explanation | Example |
| - | - | - |
| Analyze in log |  | 1 |
| Estimated particle size (log-scale) [px] |  | [9.0, 9.0] |
| Estimated particle size [px] |  | [9.0, 9.0] |
| Minimal bead brightness |  |15  |
| Separation data |  | 1.5 |


### Link
Parameters to use the Trackpy module no form trajectories of localized objects of different frames.

| Key  | Explanation | Example |
| - | - | - |
| Dark time [frames] |  | 5 |
| Dwell time stationary objects |  | 100 |
| Max displacement [px] |  | 5 |
| Max displacement fix [px] |  | 0.3 |
| Min tracking frames before drift |  | 15 |
| Min_tracking_frames |  | 65 |


### StationaryObject
Some Objects do not move. They might be attached to something. You never know. A subroutine checks it.
   
| Key  | Explanation | Example |
| - | - | - |
| Analyze fixed spots |  | 1 |
| Min distance to stationary object [px] |  | 20 |
   

### Split
Sometimes a trajectory must be cut in two. A reason can be, that the assignment was wrong.
   
 | Key  | Explanation | Example |
| - | - | - |
| Max rel median intensity step |  | 0.5 |
   

### Drift
A drift can occur during the measurement. The drift needs to be measured and subtracted from the movement in order to get the diffusion only
   
 | Key  | Explanation | Example |
| - | - | - |
| Do transversal drift correction |  | 1 |
| Drift smoothing frames |  | 20 |
| Drift rolling window size |  | 5 |
| Min particle per block |  | 40 |
   
### Old
Parameters that are unused but still there in case they turn modern again
   
| Key  | Explanation | Example |
| - | - | - |
| Min displacement [px] |  | 5 |
| Rolling window size|  | 5 |
   

### MSD
Parameters of the mean-squared-displacement fit.

   
 | Key  | Explanation | Example |
| - | - | - |
| Amount summands |  | 30 |
| Amount lagtimes auto |  | 1 |
| Amount lagtimes |  | 5 |

   
### Plot
Parameters that define what is ploted, how it looks like and where it is safed.

   
| Key  | Explanation | Example |
| - | - | - |
| Background_Show |  | 1 |
| Background_Save |  | 1 |
| Laserfluctuation_Show |  | 1 |
| Laserfluctuation_Save |  | 1 |
| MSD_fit_Show |  | 1 |
| MSD_fit_Save |  | 1 |
| Histogramm_Show |  | 1 |
| Histogramm_Save |  | 1 |
| Bins |  | 25 |
| Cutoff Size [nm] |  | 200 |
| fontsize |  | 14 |
| SaveFolder |  | "Z:\\Datenauswertung\\19_ARHCF\\" |
| SaveProperties |  | "Z:\\Datenauswertung\\19_ARHCF\\PlotProperties.json" |
| dpi |  | 120 |
  
