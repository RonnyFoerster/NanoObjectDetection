Documentation is here:

https://htmlpreview.github.io/?https://github.com/ronnyfoerster/nanoobjectdetection/blob/master/docs/_build/html/index.html



# NanoObjectDetection


# JSON settings file
## Important
- All the experimental parameters are find in a *.json file.
- The json datatype is quite strict with the format. The last entry of a key should not end whith a " , " . Booleans are written as 0 and 1.
- A path does not except " \ " instead you have to use " \\\ "

## Explenation of parameters

### Exp
Here are all experimental parameters

| Key  | Explanation | Example | Unit |
| - | - | - | - |
| NA  | numberical aperture of the detection objective | 0.13 | - |
| lambda | wavelength of the detection light (in scattering this is the illumination wavelength) | 532 | nm |
| fps | frames per seconds (fps) of the camera; required for the lagtime (1/fps) |  97.281 | Hz |
| ExposureTime | exposuretime of each raw image | 0.01028 | seconds |
| Microns_per_pixel | size of a camera pixel in object coordinate. This is given by the pixel pitch of the CCD and the magnification of the system. Be aware that the magnification written on an objective is only valid for the correct tube lens length. | 0.63 | µm/px |
| Temperature | temperature of the specimen | 295.0 | °C |
| const_Boltz | Boltzmann constant for the diffusion | 1.38e-23 | - |


### File
Here are alle the file locations of the input

| Key  | Explanation | Example | Unit |
| - | - | - | - |
| data_file_name | Complete path of the first image  | "C:\\Data\\Mona_Nissen\\40nmgold.fits" | - |
| data_folder_name | Location of the images  | "C:\\Data\\Mona_Nissen" | - |
| data_type | File format of the images | "fits", "tif_stack", "tif_series" | - |
| DefaultParameterJsonFile | If a json file is incomplete, for example due to an update, the json file is completed by the default value, whose path defined in here. | "Z:\\Datenauswertung\\19_ARHCF\\190802\\190225_default.json" | - |
| json | path of the json file (if not given the code will write it into it.) | "Z:\\Datenauswertung\\Mona_Nissen\\Au50_922fps_mainChanOnly\\190227_Au50_922fps_mainChanOnly.json" | - |

### ROI - Region of interest
Here a subregion (ROI) can be defined. This is especially helpfull for debugging, when not the entire image or all frames shall be used for the calculation in order to speed up the calculations. In addition, it can be helpfull if the image is large, but you just wanna evaluate a particle at a very defined region and time.

| Key  | Explanation | Example | Unit |
| - | - | - | - |
| UseROI | Boolean if the ROI is applied | 1 | - |
| x_min | - | 0 | - |
| x_max | - | 100 | - |
| y_min | - | 25 | - |
| y_max | - | 80 | - |
| frame_min | - | 0 | - |
| frame_max | - | 500 | - |
  

### PreProcessing
Standard image processing before the specific MSD stuff starts.
   
| Key  | Explanation | Example | Unit |
| - | - | - | - |
| Remove_CameraOffset | Boolean if the camera offset is removed | 1 | - |
| Remove_Laserfluctuation | Boolean if the laser fluctuations are removed | 1 | - |
| Remove_StaticBackground | Boolean if the static background is removed | 1 | - |
| RollingPercentilFilter | Boolean if a rolling filter is applied on the rawimages along the time axis | 0 | - |
| RollingPercentilFilter_rolling_step | Number of frames that are binned together as one. | 10 | frames |
| RollingPercentilFilter_rolling_length | Length of the rolling filter. In other words, the number of frames that are averaged. | 200 | frames |
| RollingPercentilFilter_percentile_filter | Percentile which is used for the fitlering | 80 | - |
| ClipNegativeValue | Boolean, if the negative values are set to 0. Negative values seem unnatural, but contain valuable information | 0 | - |
| Do_or_apply_data_rotation | Boolean if the rawdata is rotated | 0 | - |
| rot_angle | Angle by which the rawdata is rotated  | 0.3 | degree |


### Help
Start a routine that try to help you find the right parameters.

      
| Key  | Explanation | Example |
| - | - | - | - |
| ROI | Boolean if help with the ROI parameters is wanted | 1 | - |
| Minimal bead brightness | Boolean if help with the "Minimal bead brightness" is wanted | 1 | - |


### Find
Parameters necessary to run the Tracky module.
        
| Key  | Explanation | Example | Unit |
| - | - | - | - |
| Analyze in log | Some people like to form the log of a rawimage before proceeding | 0 | - |
| Estimated particle size | estimated particle size of an object on the camera. This is required for Trackpy to find it with high accuaray. The help routine is of use if unknown | [9.0, 9.0] | px |
| Estimated particle size (log-scale)  | as above | [9.0, 9.0] | px |
| Minimal bead brightness | Minimal brightness a bead must have in order to be classified as an object and not as noise | 15 | - |
| Separation data | Minimum distance of two identified particles. | 1.5 | px |


### Link
Parameters to use the Trackpy module no form trajectories of localized objects of different frames.

| Key  | Explanation | Example | Unit |
| - | - | - | - |
| Dark time | An object can disappear from on frame to another. A reason can be local intensity fluctuations, for example generated by large stationary objects. The dark time define how long a object can be invisible before defined as lost. | 5 | frames |
| Dwell time stationary objects | Number of frames an object must remain at it position in order to be classified as stationary. | 100 | frames |
| Max displacement | Maximum movement a particle can do between to frames before beeing classified as another particle | 5 | px |
| Max displacement fix | Maximum movement of a particle that is tested between two frames for beeing stationary . Important: this check musst happen before the drift correction, because stationary objects are independent from the drift and vice versa. | 0.3 | px |
| Min tracking frames before drift |  | 15 | frames |
| Min_tracking_frames | In order to get a good statistic, the squared deviation must be measured several times. This the trajectory lenght must have minium number of frames | 65 | frames |


### StationaryObject
Some Objects do not move. They might be attached to something. You never know. A subroutine checks it.
   
| Key  | Explanation | Example | Unit |
| - | - | - | - |
| Analyze fixed spots | Booled if the specimen is tested on stationary objects | 1 | - |
| Min distance to stationary object | Stationary objects are bad, because they are normally quite large and interfere with neighbouring particles, which distracts their localzation. Thus, a maximum distance from moving to stationary objects is defined in pixels | 20 | px |
   

### Split
Sometimes a trajectory must be cut in two. A reason can be, that the assignment was wrong.
   
 | Key  | Explanation | Example | Unit | - |
| - | - | - | - |
| Max rel median intensity step | Defines the maximum relativ intensity drop/raise are trajectory can have between to neighbouring pixels. If the value is exceeded the risk of a wrong assignment is too large. Thus the trajectory is splitted at this jump. | 0.5 | - |
   

### Drift
A drift can occur during the measurement. The drift needs to be measured and subtracted from the movement in order to get the diffusion only. The drift is measured by the assumption, that the average movement is zero. Thus a drift can only be measured if many particles are present.
   
 | Key  | Explanation | Example | Unit |
| - | - | - | - |
| Do transversal drift correction | Boolean if the drift correction considers are lamiar flow | 1 | - |
| Drift smoothing frames | Number of frames the drift is averaged over, since it assumed to change slowly | 20 | frames |
| Drift rolling window size |  | If a lamiar flow is investigated, this value gives the total number of neighbouring slices that are used for averaging.| frames |
| Min particle per block | If a lamiar flow is investigated, this is the minimum number of particles in one block | 40 | - |
      

### MSD
Parameters of the mean-squared-displacement fit.

   
 | Key  | Explanation | Example | Unit |
| - | - | - | - |
| Amount summands | Minimum number of statistically indepent values at each considered lagtime. If a particles does not fulfill this criterion, it is not evaluated further. The higher the value the better the fit.  | 30 | - |
| Amount lagtimes auto | Boolean if the number of lagtimes for the fit is defined automatically | 1 | - |
| Amount lagtimes | Amount of lagtimes used for the fit | 5 | frames |

   
### Plot
Parameters that define what is ploted, how it looks like and where it is safed.

   
| Key  | Explanation | Example | Unit |
| - | - | - | - |
| Background_Show | Display the background image | 1 | - |
| Background_Save | Save the background image | 1 | - |
| Laserfluctuation_Show | Display the laser fluctuations | 1 | - |
| Laserfluctuation_Save | Save the laser fluctuations | 1 | - |
| MSD_fit_Show | Display the MSD fit | 1 | - |
| MSD_fit_Save | Save the MSD fit | 1 | - |
| Histogramm_Show | Display the diameter histogramm | 1 | - |
| Histogramm_Save | Save the diameter histogramm | 1 | - |
| Bins | Number of bins in the historgramm, between min and maximum diameter | 25 | - |
| Cutoff Size | Allow diamters above this value are discarded | 200 | nm |
| fontsize | Plot fontsize | 14 | - |
| SaveFolder | Folder where to save it | "Z:\\Datenauswertung\\19_ARHCF\\" | - |
| SaveProperties | Properties of the plot | "Z:\\Datenauswertung\\19_ARHCF\\PlotProperties.json" | - |
| dpi | Used dpi of the plot | 120 | - |
  


### Old
Parameters that are unused but still there in case they turn modern again
   
| Key  | Explanation | Example | Unit |
| - | - | - | - |
| Min displacement |  | 5 | - |
| Rolling window size|  | 5 | - |
| data_file_extension | File format of the images. | "tif" | - |
