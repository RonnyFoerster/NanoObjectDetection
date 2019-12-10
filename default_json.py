"""
IMPORTANT:
All the experimental parameters are located in a json file.
The json datatype is quite strict with the format. The last entry of a key should not end whith a " , " . Booleans are written as 0 and 1.
A path does not except " \ " instead you have to use " \\ "

########################################################################################################################

Exp

Experimental parameters

| key: NA:
| description: numberical aperture of the detection objective
| example: 0.13
| unit: none

| key: lambda
| description: wavelength of the detection light (in scattering this is the illumination wavelength)
| example: 532
| unit: nm

| key: fps
| description: frames per seconds (fps) of the camera; required for the lagtime (1/fps)
| example: 100
| unit: Hz

| key: ExposureTime
| description: exposure time of each raw image
| example: 0.01028
| unit: seconds

| key: Microns_per_pixel
| description: size of a camera pixel in object coordinate. This is given by the pixel pitch of the CCD and the magnification of the system. Be aware that the magnification written on an objective is only valid for the correct tube lens length.
| example: 0.63
| unit: µm/px
    
| key: Temperature
| description: temperature of the specimen
| example: 295
| unit: K
   
| key: Viscocity
| description: Viscocitiy of the fluid
| example: 9.5e-16
| unit: Ns/(um^2) = 1E-12 * Ns(m^2) = 1E-15 * mPa * s

| key: const_Boltz
| description: Boltzmann constant for the diffusion
| example: 1.38e-23
| unit: m^2 * kg / (s^2 * K) = J / K
 
  
########################################################################################################################

Fiber

Here are all the fiber parameters

| key: Shape
| description: Shape of the fiber
| example: "hex"
| unit: 
    
| key: ARHCF_hex_sidelength_um
| description: If its an hexagonal ARHCF, the 6 side length are insert here
| example: [315, 315, 318, 335, 300, 338]
| unit: um
      
| key: CalcTubeDiameterOutOfSidelength
| description: Calculated the fiber diameter out of the given side length and fiber shape
| example: 1
| unit: boolean
    
| key: TubeDiameter_nm
| description: Diameter of the fiber
| example: 10000
| unit: nm        
          
########################################################################################################################
       
File

Here are alle the file locations of the input
 
| key: data_file_name
| description: Complete path of the first image
| example: "C:\\Data\\Mona_Nissen\\40nmgold.fits"
| unit: 

| key: data_folder_name
| description: Location of the images
| example: "C:\\Data\\Mona_Nissen"
| unit: 
    
 
| key: data_type
| description: File format of the images
| example: "fits", "tif_stack" or "tif_series"
| unit: 
  
| key: DefaultParameterJsonFile
| description: If a json file is incomplete, for example due to an update, the json file is completed by the default value, whose path defined in here.
| example: "Z:\\Datenauswertung\\19_ARHCF\\190802\\190225_default.json"
| unit: 
 
| key: json
| description: path of the json file (if not given the code will write it into it.)
| example: "Z:\\Datenauswertung\\Mona_Nissen\\Au50_922fps_mainChanOnly\\190227_Au50_922fps_mainChanOnly.json"
| unit: 
   
########################################################################################################################

    
Simulation

In case the data is simulated and not acquired physically the parameters can be found here

| key: SimulateData
| description: Boolean if the data shall be simulated
| example: 1
| unit: boolean

| key: DiameterOfParticles
| description: Diameter of the simulated particles
| example: 100
| unit: nm

| key: NumberOfParticles
| description: Number of created particles
| example: 42
| unit: 

| key: NumberOfFrames
| description: Number of simulated frames
| example: 420
| unit: frames

| key: mass
| description: Mass of the particles
| example: 100
| unit: 

| key: EstimationPrecision
| description: Estimation precision 
| example: !!! TODO !!!
| unit: !!! TODO !!!

| key: Max_traj_length
| description: Maximum trajectory length. Longer trajectories are cutted and treates as independent particles at different time points
| example: 300
| unit: frames
    
########################################################################################################################

    
ROI - Region of interest

Here a subregion (ROI) can be defined. This is especially helpfull for debugging, when not the entire image or all frames shall be used for the calculation in order to speed up the calculations. In addition, it can be helpfull if the image is large, but you just wanna evaluate a particle at a very defined region and time.

| key: Apply
| description: Boolean if the ROI is applied
| example: 1
| unit: boolean

| key: x_min, x_max, y_min, y_max
| description: min- / maximal value of the ROI
| example: 100
| unit: px

| key: frame_min, frame_max
| description: min- / maximal processed frame
| example: 100
| unit: frames
   
########################################################################################################################


Subsampling

A subsampling of data can be useful if the data is highly oversampled or a lower framerate of the camera is sufficent
A subsampled image is analyzed faster but information is obviously thrown away.

| key: Apply
| description: Boolean if a subsampling shall be applied
| example: 1
| unit: boolean
    
| key: fac_xy
| description: Factor of spatial subsampling.
| example: 5... means that just every fifth pixel is kept used further
| unit: px
    
| key: fac_frame
| description: Factor of temporal subsampling.
| example: 5... means that just every fifth frame is kept
| unit: frames

########################################################################################################################


PreProcessing

Standard image processing before the specific MSD stuff starts.

| key: Remove_CameraOffset
| description: Boolean if the camera offset is removed
| example: 1
| unit: boolen

| key: Remove_Laserfluctuation
| description: Boolean if the laser fluctuations are removed
| example: 1
| unit: boolen

| key: Remove_StaticBackground
| description: Boolean if the static background is removed
| example: 1
| unit: boolen
 
| key: RollingPercentilFilter
| description: Boolean if a rolling filter is applied on the rawimages along the time axis
| example: 1
| unit: boolen    
 
| key: RollingPercentilFilter_rolling_step
| description: Number of frames that are binned together as one.
| example: 10
| unit: frames    
 
| key: RollingPercentilFilter_rolling_length
| description: Length of the rolling filter. In other words, the number of frames that are averaged
| example: 100
| unit: frames    
     
| key: RollingPercentilFilter_percentile_filter
| description: Percentile which is used for the fitlering
| example: 80
| unit: %  

| key: ClipNegativeValue
| description: Boolean, if the negative values are set to 0. Negative values seem unnatural, but contain valuable information	0
| example: 1
| unit: boolean  
 
| key: Do_or_apply_data_rotation
| description: Boolean if the rawdata is rotated
| example: 1
| unit: boolean  
 
| key: rot_angle
| description: Angle by which the rawdata is rotated
| example: 0.3
| unit: degree  

########################################################################################################################


Help

Start a routine that try to help you find the right parameters.


| key: ROI
| description: Boolean if help with the ROI parameters is wanted
| example: 1
| unit: boolean 
 
| key: Bead brightness
| description: Boolean if help with the "Bead brightness" is wanted
| example: 1
| unit: boolean
    
| key: Bead size
| description: Boolean if help with the "Bead size" is wanted
| example: 1
| unit: boolean

########################################################################################################################


Find

Parameters necessary to find the particles in the image with the Trackpy module.

 
| key: Analyze in log
| description: Some people like to form the log of a rawimage before proceeding
| example: 0
| unit: boolean
    
| key: Estimated particle size
| description: estimated particle size of an object on the camera. This is required for Trackpy to find it with high accuaray. The help routine is of use if unknown		
| example: [9.0, 9.0]
| unit: px
     
| key:  Estimated particle size (log-scale)		
| description:  as above (Estimated particle size)
| example: [9.0, 9.0]
| unit: px
     
| key: Minimal bead brightness	
| description: Minimal brightness a bead must have in order to be classified as an object and not as noise
| example: 15
| unit: 
     
| key: Separation data
| description: 	Minimum distance of two identified particles
| example: 1.5
| unit: px

########################################################################################################################


Link
    
Parameters to use the Trackpy module to form trajectories of localized objects of different frames.


| key: Dark time
| description: An object can disappear from one frame to another. A reason can be local intensity fluctuations, for example generated by large stationary objects. The dark time defines how long an object can be invisible before defined as lost.
| example: 5
| unit: frames
    
| key: Dwell time stationary objects
| description: Number of frames an object must remain at its position in order to be classified as stationary.
| example: 100
| unit: frames
    
| key: Max displacement
| description: Maximum movement a particle can do between two subsequent frames before being classified as another particle
| example: 3
| unit: px
    
| key: Max displacement fix
| description: Maximum movement of a particle that is tested between two frames for being stationary. Important: this check must happen before the drift correction, because stationary objects are independent of the drift and vice versa.
| example: 0.3
| unit: px
    
| key: Min tracking frames before drift
| description: Minimum length of a trajectory before the drift correction. A too short trajectory is considered of beeing wrong assigned or very noise
| example: 15
| unit: frames
    
| key: Min_tracking_frames
| description: In order to get a good statistic, the squared deviation must be measured several times. Thus the trajectory length must have minium number of frames.
| example: 500
| unit: frames
    
########################################################################################################################

StationaryObjects

Some Objects do not move. They might be attached to something. You never know. A subroutine checks it.

| key: Analyze fixed spots
| description: Boolean weather the algorithm shall look for fixed spots
| example: 1 
| unit: boolean
    
| key: Min distance to stationary object
| description: Stationary objects are bad, because they are normally quite large and interfere with neighbouring particles, which distracts their localization. Thus, a minimum distance from moving to stationary objects is defined (in pixels)
| example: 20
| unit: px

########################################################################################################################


Split

Sometimes a trajectory must be cut in two. A reason can be, that the assignment was wrong.

| key: Max_traj_length
| description: Maximum trajectory length. Longer trajectories are cutted and treates as independent particles at different time points
| example: 500
| unit: frames
    
| key: Max_traj_length_keep_tail
| description: If a trajectory is cutted, there will be a left over. This 'tail' can be kept, so that one trajectory is longer (1), or can be discarded (0) so that all trajectories are of equal length.
| example: 1
| unit: boolean
    
| key: Max rel median intensity step
| description: Defines the maximum relative intensity drop/raise a trajectory can have between two neighbouring pixels. If the value is exceeded, the risk of a wrong assignment is too large. Thus the trajectory is splitted at this jump. 
| example: 0.5
| unit: 

| key: ParticleWithLongestTrajBeforeSplit
| description: Particle ID with the longest trajectory. The longest trajectory can be cutted into the most subtrajectories, which can be all evaluated independently, which gives something like a std/ resolution of the measurement / analysis
| example: 42 (this value is inserted by the algorithm)
| unit: ID

########################################################################################################################


Drift
 
A drift can occur during the measurement. The drift needs to be measured and subtracted from the movement in order to 
get the diffusion only. The drift is measured by the assumption, that the average movement is zero. Thus a drift can 
only be measured if many particles are present.

| key: Apply
| description: Boolean weather a drift correction is performed
| example: 1
| unit: boolean
    
| key: Do transversal drift correction
| description: Boolean if the drift correction considers are lamiar flow (1) or global drift (0)
| example: 1
| unit: 
    
| key: Drift smoothing frames
| description: Number of frames the drift is averaged over, since it assumed to change slowly
| example: 20
| unit: frames
    
| key: Drift rolling window size
| description: Lamiar flow: this value gives the total number of neighbouring slices that are used for averaging.
| example: 5
| unit:
    
| key: Min particle per block
| description: Laminar flow: This is the minimum number of particles in one block
| example: 40
| unit: 
    

########################################################################################################################

MSD

Parameters of the mean-squared-displacement fit.
 
| key: Amount summands
| description: Minimum number of statistically independent values at each considered lagtime. If a particles does not fulfill this criterion, it is not evaluated further. The higher the value the better the fit.
| example: 30
| unit: 
 
| key: Amount lagtimes auto
| description: Boolean if the number of lagtimes for the fit is defined automatically
| example: 1
| unit: boolean
     
| key: Amount lagtimes
| description: Amount of lagtimes used for the fit
| example: 5
| unit: frames
     
| key: effective_fps
| description: The framerate might be reduced due to subsampling
| example: 42
| unit: frames
     
| key: effective_Microns_per_pixel
| description: The magnification might be reduced (i.e. the pixel conversion number increased) due to subsampling
| example: 0.5
| unit: um/px
     
| key: EstimateHindranceFactor
| description: Boolean if hindrance factor shall be considered (good when particle diameter in the range of the fiber diameter)
| example: 1
| unit: boolean
     
| key: EvalOnlyLongestTraj
| description: Boolean if only the longest trajectory shall be evaluated
| example: 1
| unit: boolean
  
########################################################################################################################

Plot

Parameters that define what is ploted, how it looks like and where it is safed.
 
| key: Background_Show
| description: Show background / dark image
| example:
| unit: boolean

| key: Background_Save
| description: 
| example:
| unit: boolean

| key: Laserfluctuation_Show
| description: Show laserintensity (only valid for if many particles are present) over time
| example:
| unit: boolean

| key: Laserfluctuation_Save
| description: 
| example:
| unit: boolean

| key: MSD_fit_Show
| description: MSD over lagtime
| example:
| unit: boolean

| key: MSD_fit_Save
| description: 
| example:
| unit: boolean

| key: Histogramm_Show
| description: Diameter histogramm
| example:
| unit: boolean

| key: Histogramm_Save
| description: 
| example:
| unit: boolean

| key: Histogramm_Bins_Auto
| description: Automatically sets the number of bins in the diameter histogramm
| example: 1
| unit: boolean
    
| key: Histogramm_Bins
| description: Number of bins in the diameter histogramm
| example: 25
| unit: 
    
| key: Histogramm_min_max_auto
| description: Automatically the minimal and maximal diameter in the diameter histogramm
| example: 1
| unit: boolean
    
| key: Histogramm_min
| description: Minimal diameter in the diameter histogramm
| example: 30
| unit: nm
    
| key: Histogramm_max
| description: Maximal diameter in the diameter histogramm
| example: 100
| unit: nm
    
| key: DiamOverTraj_Show
| description: Diameter of a particle over its trajectory length
| example:
| unit: boolean
    
| key: DiamOverTraj_Save
| description: 
| example:
| unit: boolean

| key: fontsize
| description: Plot fontsize
| example: 14
| unit: 
    
| key: SaveFolder
| description: Folder where to save it
| example: "Z:\\Datenauswertung\\Mona_Nissen\\Au50_922fps_mainChanOnly"
| unit:
    
| key: save_json
| description: Boolean weather the properties are saved
| example: 1
| unit: boolean
    
| key: SaveProperties
| description: The json file is saved along with the image. This makes it reproduceable
| example: "Z:\\Datenauswertung\\Mona_Nissen\\Au50_922fps_mainChanOnly\\190305\\16_16_05_Diameter Histogramm.json"
| unit:
    
| key: dpi
| description: dpi of the saved image
| example: 300
| unit: dpi

| key: save_data2csv
| description: Boolean weather the data is saved
| example: 1
| unit: boolean



########################################################################################################################


Old
 
Parameters that are unused but still there in case they turn modern again

| key: Min displacement
| description: TODO !!!
| example:5
| unit: 

| key:Rolling window size
| description:  TODO !!!
| example: 5
| unit: 
    
| key: data_file_extension
| description: File format of the images
| example: "tif"
| unit: 
    
"""