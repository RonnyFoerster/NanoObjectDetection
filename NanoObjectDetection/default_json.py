"""
IMPORTANT:
All the experimental parameters can be found in a json file.
The json datatype is quite strict with the format. The last entry of a key should not end whith a " , " . Booleans are written as 0 and 1.
A path does not accept " \ " instead you have to use " \\ ".

##############################################################################

Exp

Experimental parameters

| key: NA:
| description: numerical aperture of the detection objective
| example: 0.13

| key: n_immersion
| description: Refractive index of the objective immersion
| example: 1.46

| key: lambda
| description: wavelength of the detection light (in scattering this is the illumination wavelength)
| example: 532
| unit: nm

| key: fps
| description: frames per seconds (fps) of the camera; required for the lagtime (1/fps)
| example: 100
| unit: Hz

| key: ExposureTime
| description: exposuretime of each raw image
| example: 0.01
| unit: seconds

| key: Microns_per_pixel
| description: size of a camera pixel in object coordinate. This is given by the pixel pitch of the CCD and the magnification of the system. Be aware that the magnification written on an objective is only valid for the correct tube lens length.
| example: 0.63
| unit: µm/px

| key: Temperature
| description: temperature of the specimen
| example: 295
| unit: K
   
| key: Viscosity
| description: Viscosity of the fluid
| example: 9.5e-16
| unit: Ns/(um^2) = 1E12 * Ns/(m^2) = 1E15 * mPa * s

| key: Viscosity_auto
| description: gets the viscosity automatically by solvent and temperature
| example: 1

| key: solvent
| description: liquid inside the channel
| example: "water"

 
##############################################################################

Fiber

Here are all the fiber parameters

| key: Mode
| description: Describes the mode in the channel
| options: "Speckle", "Gaussian", "Evanscent" (last not implemented)
   
| key: Waist
| description: Beam waist in case of gaussian mode
| example: 8.6
| unit: um
    
| key: TubeDiameter_nm
| description: Diameter of fiber channel where the particles are in (distance between parallel walls)
| example: 10000
| unit: nm        
          
##############################################################################
       
File

Here are all the file locations of the input.
 
| key: data_file_name
| description: Complete path of the first image
| example: "C:\\Data\\Mona_Nissen\\40nmgold.fits"

| key: data_folder_name
| description: Location of the images
| example: "C:\\Data\\Mona_Nissen"   
 
| key: data_type
| description: File format of the images. TIF-STACK IS HIGHLY RECOMMENDED, BECAUSE IT READS IN MUCH FASTER
| example: "fits", "tif_stack" or "tif_series"
  
| key: use_num_frame
| description: umber of 2d tif images that are read in (Only valid for tif-series!).
| example: 100
 
| key: DefaultParameterJsonFile
| description: If a json file is incomplete, for example due to an update, the json file is completed by the default value, whose path defined in here.
| example: "Z:\\Datenauswertung\\19_ARHCF\\190802\\190225_default.json"
 
| key: json
| description: path of the json file (if not given the code will write it into it.)
| example: "Z:\\Datenauswertung\\Mona_Nissen\\Au50_922fps_mainChanOnly\\190227_Au50_922fps_mainChanOnly.json"
         
##############################################################################
   
    
Logger

This is the logger (shows all the infos, warning and errors from the code)
 
| key: level
| description: Gives the minium level of the display loggers an
| example: "debug", "info", "warning", "error"
    
##############################################################################

    
ROI - Region of interest

Here a subregion (ROI) can be defined. This is especially helpfull for debugging, when not the entire image or all frames shall be used for the calculation in order to speed up the calculations. In addition, it can be helpfull if the image is large, but you just wanna evaluate a particle at a very defined region and time.

| key: Apply
| description: Boolean if the ROI is applied
| example: 1

| key: Save
| description: Boolean if the ROI is saved
| example: 1

| key: x_min, x_max, y_min, y_max
| description: min- / maximal value of the ROI
| example: 100
| unit: px

| key: frame_min, frame_max
| description: min- / maximal processed frame
| example: 100
| unit: frames
   
##############################################################################

PreProcessing

Standard image processing before the specific MSD stuff starts.

| key: Remove_CameraOffset
| description: Boolean if the camera offset is removed
| example: 1

| key: Remove_Laserfluctuation
| description: Boolean if the laser fluctuations are removed
| example: 1

| key: Remove_StaticBackground
| description: Boolean if the static background is removed
| example: 1
    
| key: EnhanceSNR
| description: Convolved with the PSF (for low SNR images, PSF approximated by gauss)
| example: 1    
        
| key: KernelSize
| description: Kernelsize of gauss filtering
| example: "auto" or fixed value
| unit: Pixel
        
| key: ZNCC_min
| description: Threshold of the Zero Normalized Cross Correlation to identify paticle (0 to 1)
| example: 0.7

##############################################################################


Help

Start a routine that try to help you find the right parameters.

| key: ROI
| description: Boolean if help with the ROI parameters is wanted
| example: 1
 
| key: Bead brightness
| description: Boolean if help with the "Bead brightness" is wanted
| example: 1
    
| key: Bead size
| description: Boolean if help with the "Bead size" is wanted
| example: 1
    
| key: TryFrames
| description: Number of frames the parameter estimation is run for bead brightness and size
| example: 10
| unit: frames
    
| key: Separation
| description: Boolean if the separation (distance) between two objects is set automatically
| example: 1
    
| key: GuessLowestDiameter_nm
| description: Smallest expected diameter. Required to calculated the minimum separation distance.
| example: 20
| unit: nm
    
| key: drift
| description: Boolean if the drift correction is on
| example: 1

##############################################################################


Find

Parameters necessary to find the particles in the image with the Tracky module.
    
| key: tp_diameter
| description: estimated particle size of an object on the camera. This is required for Trackpy to find it with high accuaray. The help routine is of use if unknown		
| example: [9.0, 9.0]
| unit: px
     
| key: tp_minmass
| description: Minimal brightness a bead must have in order to be classified as an object and not as noise
| example: 15
  
| key: SaturatedPixelValue
| description: Pixel value at which a particle is saturated.
| example: 64000 or "auto" (finds the max value automatically and saves it here)
     
| key: tp_separation
| description: 	Minimum distance of two identified particles
| example: 1.5
| unit: px
     
| key: tp_percentile
| description: Features must have a peak brighter than pixels in this percentile. This helps eliminate spurious peaks.
| example: 20
| unit: %

##############################################################################


Link
    
Parameters to use the Trackpy module to form trajectories of localized objects of different frames.

| key: Dark time
| description: An object can disappear from on frame to another. A reason can be local intensity fluctuations, for example generated by large stationary objects. The dark time define how long a object can be invisible before defined as lost.
| example: 5
| unit: frames
    
| key: Dwell time stationary objects
| description: Number of frames an object must remain at it position in order to be classified as stationary.
| example: 100
| unit: frames
    
| key: Max displacement
| description: Maximum movement a particle can do between to frames before beeing classified as another particle
| example: 3
| unit: px
    
| key: Max displacement fix
| description: Maximum movement of a particle that is tested between two frames for beeing stationary. Important: this check musst happen before the drift correction, because stationary objects are independent  from the drift and vice versa.
| example: 0.3
| unit: px
    
| key: Min tracking frames before drift
| description: Minimum length of a trajectory before the drift correction. A too short trajectory is considered of beeing wrong assigned or very noise
| example: 15
| unit: frames
    
| key: Min_tracking_frames
| description: In order to get a good statistic, the squared deviation must be measured several times. This the trajectory lenght must have minium number of frames.
| example: 500
| unit: frames
    
##############################################################################

StationaryObjects

Some Objects do not move. They might be attached to something. You never know. A subroutine checks it.

| key: Analyze fixed spots
| description: Boolean weather the algorithm shall look for fixed spots
| example: 1 
    
| key: Min distance to stationary object
| description: Stationary objects are bad, because they are normally quite large and interfere with neighbouring particles, which distracts their localzation. Thus, a maximum distance from moving to stationary objects is defined in pixels bright particle cluster)
| example: 20
| unit: px

##############################################################################


Split

Sometimes a trajectory must be cut in two. A reason can be, that the assignment was wrong.

| key: Max_traj_length
| description: Maximum trajectory length. Longer trajectories are cutted and treates as independent particles at different time points
| example: 500
| unit: frames
    
| key: Max_traj_length_keep_tail
| description: If a trajectory is cutted, there will be a left over. This 'tail' can be kept, so that one trajectore is longer (1), or can be discarded (0) so that all trajectories are of equal length.
| example: 1
    
| key: IntensityJump
| description: if trajectory shall be splitted at to high intensity jumps 
| example: 1

| key: Max rel median intensity step
| description: Defines the maximum relativ intensity drop/raise are trajectory can have between to neighbouring pixels.  If the value is exceeded the risk of a wrong assignment is too large. Thus the trajectory is splitted at this jump. 
| example: 0.5

| key: ParticleWithLongestTrajBeforeSplit
| description: Particle ID with the longest trajectory. The longest trajectory can be cutted into the most subtrajectories, which can be all evaluated independently, which gives something like a std/ resolution of the measurement / analysis
| example: 42 (this value is inserted by the algorithm)

##############################################################################


Drift
 
A drift can occur during the measurement. The drift needs to be measured and subtracted from the movement in order to 
get the diffusion only. The drift is measured by the assumption, that the average movement is zero. Thus a drift can 
only be measured if many particles are present.

| key: Apply
| description: Boolean weather a drift correction is performed
| example: 1
    
| key: Do transversal drift correction
| description: Boolean if the drift correction considers are lamiar flow (1) or global drift (0)
    
| key: Drift smoothing frames
| description: Number of frames the drift is averaged over, since it assumed to change slowly
| example: 20
| unit: frames
    
| key: Drift rolling window size
| description: Lamiar flow: this value gives the total number of neighbouring slices that are used for averaging.
| example: 5
    
| key: Min particle per block
| description: Laminar flow: This is the minimum number of particles in one block
| example: 40
| unit: particles
    

##############################################################################

MSD

Parameters of the mean-squared-displacement fit.
 
| key: Amount summands
| description: Minimum number of statistically indepent values at each considered lagtime. If a particles does not fulfill this criterion, it is not evaluated further. The higher the value the better the fit.
| example: 30
 
| key: Amount lagtimes auto
| description: Boolean if the number of lagtimes for the fit is defined automatically
| example: 1

| key: Estimate_X
| description: Choose mode to calculate reduced localication accuracy (x), required to choose the optimal lagtimes in the MSD fit
| options: "Exp", "Theory"
     
| key: "lagtimes_min" and "lagtimes_max"
| description: Minimal and maximal considered lagtime for the MSD fit
| example: 1
| unit: frames
     
| key: EstimateHindranceFactor
| description: Boolean if hindrance factor shall be considered (good when particle diameter in the range of the fiber diameter)
| example: 1
     
| key: EvalOnlyLongestTraj
| description: Boolean if only the longest trajectory shall be evaluated
| example: 1

| key: CheckBrownianMotion
| description: Boolean if each trajectory is checked for pure Brownian motion by a Kolmogorow-Smirnow-Test
| example: 1
  
##############################################################################

Plot

Parameters that define what is ploted, how it looks like and where it is safed.
 
| key: Background
| description: Show background / dark image
| example: 1

| key: Laserfluctuation
| description: Show laserintensity (only valid for if many particles are present) over time
| example: 1

| key: MSD_fit
| description: MSD over lagtime
| example: 1

| key: Histogramm
| description: Diameter histogramm
| example: 1

| key: Histogramm_abs_or_rel
| description: weighting of the histogramm (abs: no weighting; rel - weight by trajectory length)
| option: "both", "abs", "rel"

| key: Histogramm_Fit_1_Particle
| description: Fit Histogram assuming (Ultra-)uniform particle specimen
| example: 1

| key: Histogramm_time
| description: Shows time dependent histogram
| example: 1
  
| key: Histogramm_Bin_size_nm
| description: Width of each bin in nm
| example: 25
  
| key: Histogramm_Bins
| description: Number of bins in the diameter histogramm (generated by code!)
| example: 25
    
| key: Histogramm_min_max_auto
| description: Automatically the minimal and maximal diameter in the diameter histogramm
| example: 1
    
| key: Histogramm_min and Histogramm_max
| description: Minimal and maximial diameter in the diameter histogramm
| example: 30
| unit: nm
    
| key: DiamOverTraj_Show
| description: Diameter of a particle over its trajectory length
| example: 1

| key: DiameterPDF
| description: Plot diameter probability density function (PDF)
| example: 1

| key: DiamOverTraj
| description: Plot diameter probability density function (PDF)
| example: 1

| key: DiamOverTraj_UseRawMass
| description: Definese which mass is used for the false colour in DiamOverTraj
| options: "mean", "median", "max"

| key: fontsize
| description: Plot fontsize
| example: 14
    
| key: SaveFolder
| description: Folder where to save it
| example: "Z:\\Datenauswertung\\Mona_Nissen\\Au50_922fps_mainChanOnly"
| option: "auto" choose the file automatically
    
| key: save_json
| description: Boolean weather the properties are saved
| example: 1
    
| key: dpi
| description: dpi of the saved image
| example: 300
| unit: dpi

| key: save_data2csv
| description: Boolean weather the data is saved
| example: 1



"""