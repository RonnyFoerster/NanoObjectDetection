# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: foersterronny
"""


"""
******************************************************************************
Some settings
"""
#pd.set_option('display.max_columns',20) # I'm just using this to tell my Spyder-console to allow me to 
# see up to 20 columns of a dataframe (instead of a few only by default)


# In[1] 
"""
******************************************************************************
Declaration of variables
"""

"""
Analytics-Parameters:
"""
#mpl.rc('image', cmap='jet') # That's the old Python's standard colormap. I prefer this one over the current standard

'''
Physical--Parameters:
'''
visc_water = 9.5e-16 # Viscocity of water in [N*s/µm^2] at T = 295 K
const_Boltz = 1.38e-23 # Boltzmann-constant

'''
Data-Parameters:
'''
#data_folder_name = r'C:\Users\f22328\Desktop\170616_Harvard_Mona_Yoav\40nm' # location of image-data
data_folder_name = r'C:\Users\foersterronny\Dropbox\Dropbox\FaserProjekt\Daten\40nm'
data_file_extension = 'tif' # Extension of the image-file-type that is to be opened

'''
Measurement-Conditions:
'''
microns_per_pixel = 0.63 # depends on camera and optical magnification 
# that's the correct value for Monas Harvard measurements!!
frames_per_second = 97.281 # depends on camera settings
# that's the correct value for Monas Harvard measurements!!
temp_water = 295 # Temperature of water in fiber

'''
Image-Handling-Parameters
'''
x_min_global = 50 # Decreases size of image that is taken for particle-tracking 
x_max_global = 2000 # Decreases size of image that is taken for particle-tracking 
y_min_global = 50 # Decreases size of image that is taken for particle-tracking
y_max_global = 110 # Decreases size of image that is taken for particle-tracking
rot_angle = 0.3 # Rotation-angle: Image is rotated that much for analysis (e.g. record wasn't perfectly aligned)

'''
Tracking-and Linking-Parameters:
'''
estimated_particle_size = (9,9) # Roughly the particle size, given in pixels, odd!!!! integer-type data. 
# Preffered to over-estimate rather than under-estimate. 
# If expressed in tuple-form, x- and y-sizes can be addressed seperately (if non-spherical particles are observed)
# If chosen too small, positions are binned to full integers statistically.
# That messes up the analysis!
minimal_brightness_data = 5000 # Minimal brightness that is taken into account for spot to be regarded as a particle.
# This value might have to be reduced when less bright points are measured
max_displacement = 5 # Maximal allowed displacement of a particle in between two frames
# If exceeded, tracking assumes to be regarding two different particles
separation_data = 3 # Minimal separation of two particles to be regarded as individual
# take it out completely if default should be used (twice particle diameter)
dark_time = 10 # Describes how many frames a particle might not be seen in order to be still recognized as the same one
min_tracking_frames = 10;  # Describes the minimum duration of an trajectory

'''
Drift-Correction-Parameters:
'''
Do_transversal_drift_correction = 1; #Decide wether global (0) or y-dependent/ transversal drift correction 
min_tracking_frames_before_drift = 10; # That's the minimum trajectory length that's still allowed to be taken 
#into account until and including drift-correction
drift_smoothing_frames = 5 # For drift-correction this many (forward) frames are used to smooth trajectory
max_rel_median_intensity_step = 3 # define the maximum allowed step in intensity-jump between median-filtered-frames
rolling_window_size = 5 # in drift-calculation: use blocks above and below for averaging (more particles make drift correction better)
# e.g. 2 means y subarea itself and the two above AND below
min_particle_per_block = 30 # Amounts of particles in "block" for y-sub-area-depending drift-correction



