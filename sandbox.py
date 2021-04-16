# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:45:19 2020

@author: foersterronny
"""
import numpy as np # library for array-manipulation
import matplotlib.pyplot as plt # libraries for plotting
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from pdb import set_trace as bp #debugger
from joblib import Parallel, delayed
import multiprocessing

import NanoObjectDetection as nd

## In[]
def RollingMedianFilter_main(image, window = 11):
    num_cores = multiprocessing.cpu_count()
    
    num_lines = image.shape[1]
    
    inputs = range(num_lines)

    print("start median background filter - parallel")
    background_list = Parallel(n_jobs=num_cores)(delayed(RollingMedianFilter)(image[:,loop_line,:].copy(), window) for loop_line in inputs)
    print("finished median background filter - parallel")

    background = np.asarray(background_list)
    
    background = np.swapaxes(background, 0, 1)
    
    return background

#here is the file for playing with new functions, debugging and so on
def RollingMedianFilter(image, window = 11):
    print("start median background filter")
    
    # check if window is odd
    if (window%2) == 0:
        raise ValueError("window size has to be odd")
    
    # window in one direction
    a = int((window-1)/2)
    
    first_frame = a
    last_frame = image.shape[0] - a
    
    valid_frames = np.arange(first_frame, last_frame)
    
    background = np.zeros_like(image, dtype = "double")
    
    #frame that does not exceed the winodw limit
    for loop_frame in valid_frames:
        print("loop_frame : ", loop_frame)
        # print(loop_frame)
        # background[loop_frame, : , :] = np.median(image[loop_frame-a:loop_frame+a, :, :], axis = 0)
        
        image_loop = image[loop_frame-a:loop_frame+a, :]

        min_percentile = int(0.4*image_loop.shape[0])
        max_percentile = int(0.6*image_loop.shape[0])
            
        background[loop_frame, :] = np.mean(np.sort(image_loop,axis = 0)[min_percentile:max_percentile,:],axis = 0)
        
        # background[loop_frame, :] = np.median(image[loop_frame-a:loop_frame+a, :], axis = 0)
        
    # handle frames at the beginning
    # background[:first_frame, : , :] = background[first_frame, : , :]
    background[:first_frame, :] = background[first_frame, :]
    
    # handle frames at the end
    # background[last_frame:, : , :] = background[last_frame, : , :]
    background[last_frame:, :] = background[last_frame, :]
    
    
    return background



def NewBGFilter(img):
    #checks the background if the histogramm is normal distributed by kolmogorow
    import scipy
    
    img = img / 16
    
    size_y = img.shape[1]
    size_x = img.shape[2]
    
    img_test = np.zeros([size_y,size_x])
    img_mu  = np.zeros([size_y,size_x])
    img_std = np.zeros([size_y,size_x])
    
    for loop_y in range(0, size_y):
        print(loop_y)
        for loop_x in range(0, size_x):
            test = img[: ,loop_y, loop_x]
            mu, std = scipy.stats.norm.fit(test)
            [D_Kolmogorow, _] = scipy.stats.kstest(test, 'norm', args=(mu, std))
    
            img_test[loop_y, loop_x] = D_Kolmogorow
            img_mu[loop_y, loop_x] = mu
            img_std[loop_y, loop_x] = std
    
    return img_test, img_mu, img_std



def TestSlider():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    
    x = list(range(0,11))
    y = [10] * 11
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(left = 0.1, bottom = 0.35)
    p, = plt.plot(x,y, linewidth = 2, color = 'blue')
    
    plt.axis([0, 10, 0, 100])
    
    axSlider1 = plt.axes([0.1, 0.2, 0.8, 0.05])
    
    slder1 = Slider()
    
    
    plt.show()



def TryTextBox():
    from matplotlib.widgets import TextBox
    
    fig = plt.figure(figsize = [5, 5], constrained_layout=True)

    gs = GridSpec(2, 1, figure=fig)
    ax_gs = fig.add_subplot(gs[0, 0])
    
    textbox_raw_min = TextBox(ax_gs , "x min: ", initial = "10")    

    ax_plot = fig.add_subplot(gs[1, 0])
    t = np.arange(-2.0, 2.0, 0.001)
    s = t ** 2
    initial_text = "t ** 2"
    l, = ax_plot.plot(t, s, lw=2)


    def PrintASDF(text):
        ax_plot.set_xlim([0.5])
    
    textbox_raw_min.on_submit(PrintASDF)

    return textbox_raw_min


def TryTextBox2():
    #https://matplotlib.org/3.1.1/gallery/widgets/textbox.html
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox

    fig = plt.figure(figsize = [5, 5], constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig)
    ax_plot = fig.add_subplot(gs[1, 0])
    
    plt.subplots_adjust(bottom=0.2)
    t = np.arange(-2.0, 2.0, 0.001)
    s = t ** 2
    initial_text = "t ** 2"
    l, = plt.plot(t, s, lw=2)
    
    
    def submit(text):
        t = np.arange(-2.0, 2.0, 0.001)
        print(text)
        ydata = eval(text)
        l.set_ydata(ydata)
        ax_plot.set_ylim(np.min(ydata), np.max(ydata))
        plt.draw()
    
    ax_gs = fig.add_subplot(gs[0, 0])
    textbox_raw_min = TextBox(ax_gs , "x min: ", initial = "10")    
    textbox_raw_min.on_submit(submit)
    
#    axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
#    text_box = TextBox(axbox, 'Evaluate', initial=initial_text)
#    text_box.on_submit(submit)
    
    plt.show()

    return textbox_raw_min



""" 
###############################################################################
Mona's section 
###############################################################################
""" 
    
def DiameterOverTrajLengthColored(ParameterJsonFile, sizes_df_lin, 
                                  color_by='mass', use_log=False,
                                  save_plot = True):
    """ plot (and save) calculated particle diameters vs. the number of frames
    where the individual particle is visible (in standardized format) 
    and color it by a property of choice
    """
        
    import NanoObjectDetection as nd

    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    
    if Histogramm_min_max_auto == 1:
        histogramm_min = np.round(np.min(sizes_df_lin.diameter) - 5, -1)
        histogramm_max = np.round(np.max(sizes_df_lin.diameter) + 5, -1)
    else:
        histogramm_min = None
        histogramm_max = None 
    
    histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")
    
    my_title = "Particle size over tracking time (colored by {}".format(color_by)
    if use_log==True:
        my_title = my_title + " in log scale)"
    else:
        my_title = my_title + ")"
    my_ylabel = "Diameter [nm]"
    my_xlabel = "Trajectory length [frames]"
    
    plot_diameter = sizes_df_lin["diameter"]
    plot_traj_length = sizes_df_lin["traj length"]
    
    if use_log==True:
        plot_color = np.log10(sizes_df_lin[color_by])
    else:
        plot_color = sizes_df_lin[color_by]
    
    x_min_max = nd.handle_data.Get_min_max_round(plot_traj_length,2)
    x_min_max[0] = 0
    y_min_max = nd.handle_data.Get_min_max_round(plot_diameter,1)
    
    
    plt.figure()
    plt.scatter(plot_traj_length, plot_diameter, c=plot_color, cmap='viridis')
    plt.title(my_title)
    plt.xlabel(my_xlabel)
    plt.ylabel(my_ylabel)
    plt.xlim([x_min_max[0], x_min_max[1]])
    plt.ylim([y_min_max[0], y_min_max[1]])
    plt.colorbar()

 
    if save_plot == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "DiameterOverTrajLength",
                                       settings, data = sizes_df_lin)



def DiameterPDF_transparent(ParameterJsonFile, sizes_df_lin, histogramm_min = None, 
                            histogramm_max = None, Histogramm_min_max_auto = 0, 
                            binning = None):
    """ calculate and plot the diameter probability density function of a
    particle ensemble as the sum of individual PDFs - and plot the individual 
    PDFs as well!
    
    NB: each trajectory is considered individually, the tracklength determines
        the PDF widths
    
    assumption: 
        relative error = std/mean = sqrt( 2*N_tmax/(3*N_f - N_tmax) )
        with N_tmax : number of considered lagtimes
             N_f : number of frames of the trajectory (=tracklength)

    Parameters
    ----------
    ParameterJsonFile : TYPE
        DESCRIPTION.
    sizes_df_lin : TYPE
        DESCRIPTION.
    histogramm_min : TYPE, optional
        DESCRIPTION. The default is None.
    histogramm_max : TYPE, optional
        DESCRIPTION. The default is None.
    Histogramm_min_max_auto : TYPE, optional
        DESCRIPTION. The default is None.
    binning : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    prob_inv_diam
    diam_grid
    ax
    """
    import NanoObjectDetection as nd
    from scipy.stats import norm
    # sns.set(style="dark")
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DiameterPDF_Show = settings["Plot"]['DiameterPDF_Show']
    DiameterPDF_Save = settings["Plot"]['DiameterPDF_Save']
    PDF_min = settings["Plot"]['PDF_min']
    PDF_max = settings["Plot"]['PDF_max']
    
    # calculate mean and std of the ensemble (inverse!) and the grid for plotting
    diam_inv_mean, diam_inv_std = nd.CalcDiameter.StatisticOneParticle(sizes_df_lin)
    diam_grid = np.linspace(PDF_min,PDF_max,10000) # prepare grid for plotting
    diam_grid_inv = 1/diam_grid
    
    # calculate inverse diameters and estimates of their std
    inv_diam,inv_diam_std = nd.CalcDiameter.InvDiameter(sizes_df_lin, settings)
    num_trajectories = len(sizes_df_lin) 
    
    fig,ax = plt.subplots()
    
    prob_inv_diam = np.zeros_like(diam_grid_inv)
    # loop over all individual diameters, calculate their PDFs and add them up
    for index, (loop_mean, loop_std) in enumerate(zip(inv_diam,inv_diam_std)):
        my_pdf = norm(loop_mean,loop_std).pdf(diam_grid_inv)
        my_pdf = my_pdf / np.sum(my_pdf) # normalization
        
        ax.fill_between(diam_grid,my_pdf,alpha=0.3)
        
        prob_inv_diam = prob_inv_diam + my_pdf    

    # normalize
    prob_inv_diam = prob_inv_diam / np.sum(prob_inv_diam)
    
    # instantiate a second axes that shares the same x-axis
    ax2 = ax.twinx()  
    mycol = 'tab:blue'
    ax2.plot(diam_grid,prob_inv_diam, color=mycol)
    
    # configure plot settings
    ax2.set_ylabel('Ensemble probability [a.u.]', color=mycol)
    ax2.tick_params(axis='y', labelcolor=mycol)
    
    if Histogramm_min_max_auto == 1:
        PDF_min, PDF_max = nd.visualize.GetCI_Interval(prob_inv_diam, diam_grid, 0.999)
        #histogramm_min = 0
    else:
        PDF_min, settings = nd.handle_data.SpecificValueOrSettings(PDF_min, settings, 
                                                                   "Plot", "PDF_min")
        PDF_max, settings = nd.handle_data.SpecificValueOrSettings(PDF_max, settings, 
                                                                   "Plot", "PDF_max")
    ax.set(title="Trajectories: {:3.0f}".format(num_trajectories),
           xlim = [PDF_min, PDF_max], xlabel = "Diameter [nm]", 
           ylabel = "Individual probabilities [a.u.]",
           ylim=[0, 11.5*np.max(prob_inv_diam)])
    ax2.set(ylim = [0, 1.05*np.max(prob_inv_diam)])
    
    # use scientific tick notation
    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax.yaxis.set_major_formatter(formatter) 
    ax2.yaxis.set_major_formatter(formatter) 
        
    fig.tight_layout()
    
    
    if DiameterPDF_Save == True:
        save_folder_name = settings["Plot"]["SaveFolder"]
        
        settings = nd.visualize.export(save_folder_name, "Diameter_Probability", settings,
                                       data = sizes_df_lin, ShowPlot = DiameterPDF_Show)
        
        data = np.transpose(np.asarray([diam_grid, prob_inv_diam]))
        nd.visualize.save_plot_points(data, save_folder_name, 'Diameter_Probability_Data')
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings)
    
    return prob_inv_diam, diam_grid, ax










# def AnimateTracksOnRawData(t2_long,rawframes_ROI,settings,max_frm=100,frm_start=0,gamma=1.0):
#     """ animate trajectories on top of raw data
    
#     to do:
#         - compare with functions below...!!

#         - reduce ROI to actually necessary section
#         - cut history of tracks for frames < frm_start
#         - check if trajectory data and rawframes match (?)
#         - make starting frame free to choose
#         - implement gamma transform that does not blow up the file size by a factor of 10!!
#         - choose traj. color by length/mass/size/...?
#         - convert from px to um (optionally)
# """
#     rawframes_gam = rawframes_ROI.copy()
#     if gamma != 1.0:
#         rawframes_gam = 2**16 * (rawframes_gam/2**16)**gamma # for 16bit images
#         rawframes_gam = np.uint16(rawframes_gam)
        
#     fps = settings["Exp"]["fps"]    
#     # get rawframes' dimensions
#     _,y_len_raw,x_len_raw = rawframes_ROI.shape
    
#     # prepare the data
#     trajdata = t2_long.copy() 
#     frm0 = trajdata.frame.min()
#     frmMax = trajdata.frame.max()
#     amnt_f = frmMax - frm0 + 1
#     if amnt_f > max_frm:
#         amnt_f = max_frm
#         # track = track[track.frame <= amnt_f]  
    
#     # get trajectory dimensions for plotting (and add 15px to have it more centered)
#     xMin = int(trajdata.x.min() - 15)
#     if xMin < 0:
#         xMin = 0
#     xMax = int(trajdata.x.max() + 15)
#     if xMax > x_len_raw:
#         xMax = x_len_raw
#     x_len = xMax - xMin
    
#     # make DataFrame a bit slimmer
#     trajdata = trajdata.set_index(['frame'])
#     trajdata = trajdata.drop(columns=['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'abstime'])
    
#     # prepare plotting
#     fig,ax = plt.subplots(1,figsize=plt.figaspect(y_len_raw/x_len))
#     cm_prism = plt.get_cmap('prism')
    
#     raw_img = ax.imshow(rawframes_gam[frm0,:,xMin:xMax], cmap='gray', aspect="equal",
#                         animated=True)
#     ax.set(xlabel='x [px]', ylabel='y [px]',
#            ylim=[y_len_raw-1,0] ) # invert limits of y-axis!
#     ax_t = ax.twiny()
#     ax_t.set(xlim=[xMin,xMax-1])#,ylim=[70,0])
#     # initialize plots for all particles in the whole video
#     trajplots = [ax_t.plot([],[],'-',color=cm_prism(part_id),
#                            animated=True
#                            )[0] for part_id in trajdata.particle.unique()]
    
#     # heading = ax.annotate("", (5,10), animated=True,
#     #                       bbox=dict(boxstyle="round", fc="white", alpha=0.3, lw=0) )

#     # fig.tight_layout() 
    
#     # def init_tracks():
#     #     raw_img.set_data(rawframes_gam[0,:,:])
        
#     #     for tplot in trajplots:
#     #         tplot.set_data([],[])
        
#     #     # heading.set_text("")
#     #     # ax.set(title='frame: {}, time: {}s'.format(frm_start,time_start))
#     #     # return raw_img, trajplots #,heading
    
#     def update_frame(frmN, rawframes_gam, raw_img, trajdata, trajplots): #
#         frm = frmN + frm0 # if frm0!=0, we have to add it as offset
        
#         raw_img.set_data(rawframes_gam[frm,:,xMin:xMax])
        
#         # update all trajectory plots individually
#         for traj,tplot in zip(trajdata.groupby('particle'),trajplots): 
#             # get DataFrame of an individual particle
#             _,trajdf = traj # omit the particle ID
            
#             # check if current frame adds a trajectory point
#             if frm in trajdf.index: 
#                 tplot.set_data(trajdf.loc[:frm+1].x.values, trajdf.loc[:frm+1].y.values) 
        
#         time = 1000*frm/fps # ms
#         # heading.set_text("frame: {}\ntime: {:.1f} ms".format(frm,time))
#         ax.set_title("frame: {}, time: {:.1f} ms".format(frm,time))
        
#         return raw_img, trajplots#, heading 
    
#     traj_ani = FuncAnimation(fig, update_frame, #init_func=init_tracks, 
#                              frames=amnt_f, 
#                              fargs=(rawframes_gam, raw_img, trajdata, trajplots),
#                              interval=200, blit=False, repeat=False)
#     return traj_ani

# """ why are not all trajectories plotted? """


# # anim2 = AnimateTracksOnRawData(t2_long,rawframes_ROI,settings)
# # anim2.save('Au50_raw+tracks_1000frames.mp4')


def AnimateSingleTrackOnRawData(t_1particle,rawframes_ROI,settings,frm_max=200):#, gamma=0.5):
    """ animate a single trajectory on top of cropped (!) raw data
    
    to do:
        - crop the rawimage correctly (!!)
        - implement gamma transform that does not blow up the file size by a factor of 10!!
        - choose traj. color by length/mass/size/...?
    """
    trajdata = t_1particle.copy() 
    
    # check if only 1 particle is contained
    if trajdata.particle.nunique() != 1:
        print('Trajectories of more than 1 particle are contained. \nThe longest is selected for animation.')
        tlengths = trajdata.groupby("particle").size() # get the tracklengths of all contained particles
        part_ID = tlengths[tlengths==tlengths.max()].index[0] # select the longest track and get its ID
        trajdata = trajdata[trajdata.particle == part_ID] # filter out all other tracks
    
    rawfrms = rawframes_ROI.copy()
    amnt_f,_ = trajdata.shape
    if amnt_f > frm_max:
        amnt_f = frm_max # fix a max. number of frames for the video
    frm_0 = trajdata.frame.min() # get the first frame to be displayed
    fps = settings["Exp"]["fps"]

    # y is probably not necessary to be cropped
    # y_min = trajdata.y.min()
    # y_max = trajdata.y.max()
    # include some more pixels left and right to get a nice view of the whole particle
    x_crop0 = trajdata.x.min() - 20 
    x_crop1 = trajdata.x.max() + 20
    # check if one of the limits exceeds the rawframes dimensions
    if x_crop0 < 0:
        x_crop0 = 0
    if x_crop1 > rawfrms.shape[2]:
        x_crop1 = None # with this as index, also the last value of array will be included
    x_ext = x_crop1 - x_crop0    
    y_ext = rawfrms.shape[1] # image extend in y-direction

    
    """ to be continued... 
    
    attention: 2 different "ax" objects (= artists?!) probably become necessary here
    due to different x and y labeling for cropped rawframes
    
    or can this be handled via xlim, ylim??
    """
    # ax2 = ax1.twiny() 
    
    fig,ax = plt.subplots(1,figsize=plt.figaspect(y_ext/x_ext))
    ax.set_xlim(x_crop0,x_crop1)
    ax.set_ylim(y_ext-1,0) # invert for correct display!
    ax.set(xlabel='x [px]', ylabel='y [px]')
    # fig.tight_layout() 
    cm_viri = plt.get_cmap('viridis')
    
    # prepare the data
    trajdata = trajdata.set_index(['frame'])
    meanmass = trajdata.mass.mean()
    trajdata = trajdata.drop(columns=['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'abstime'])
    
    # raw_img = ax.imshow(rawfrms[frm_0,:,:], cmap='gray', aspect="equal",
    #                     animated=True)
    # raw_img = ax.imshow(rawfrms[frm_0,:,x_crop0:x_crop1], cmap='gray', aspect="equal",
    #                     animated=True)

    trajplot = ax.plot([],[],'-',color=cm_viri(meanmass), animated=True)[0]
    
    
    def init_track():
        # raw_img.set_data(rawfrms[frm_0,:,:])
        trajplot.set_data([],[])
        
    
    def update_frame(frm): #,trajdata): #,trajplots):
        # raw_img.set_data(rawfrms[frm,:,:])
        
        trajplot.set_data(trajdata.loc[:frm+1].x, trajdata.loc[:frm+1].y)
        
        time = 1000*frm/fps # ms
        # heading.set_text("frame: {}\ntime: {:.1f} ms".format(frm,time))
        ax.set_title("frame: {}, time: {:.1f} ms".format(frm,time))
        
        return trajplot #raw_img, trajplot
    
    traj_ani = FuncAnimation(fig, update_frame, init_func=init_track, 
                             frames=amnt_f, #fargs=(trajdata),#, trajplots),
                             interval=70, blit=False, repeat=False)
    return traj_ani



def AnimateTracksOnRawData(t2_long,rawframes_ROI,settings,frm_start=0):#, gamma=0.5):
    """ animate trajectories on top of raw data
    
    to do:
        - make starting frame free to choose
        - implement gamma transform that does not blow up the file size by a factor of 10!!
        - choose traj. color by length/mass/size/...?
"""
    rawframes_gam = rawframes_ROI.copy()
    # rawframes_gam = 2**16 * (rawframes_gam/2**16)**gamma # for 16bit images
    # rawframes_gam = np.uint16(rawframes_gam)
    
    amnt_f,y_len,x_len = rawframes_ROI.shape
    fps = settings["Exp"]["fps"]
    
    fig,ax = plt.subplots(1,figsize=plt.figaspect(y_len/x_len))
    cm_prism = plt.get_cmap('prism')
    
    # prepare the data
    trajdata = t2_long.copy() 
    trajdata = trajdata.set_index(['frame'])
    trajdata = trajdata.drop(columns=['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'abstime'])
    
    
    raw_img = ax.imshow(rawframes_gam[0,:,:], cmap='gray', aspect="equal",
                        animated=True)
    # heading = ax.annotate("", (5,10), animated=True,
    #                       bbox=dict(boxstyle="round", fc="white", alpha=0.3, lw=0) )
    # initialize plots for all particles in the whole video
    trajplots = [ax.plot([],[],'-',color=cm_prism(part_id),
                         animated=True
                         )[0] for part_id in trajdata.particle.unique()]
    
    ax.set(xlabel='x [px]', ylabel='y [px]',
           xlim=[0,x_len-1], ylim=[y_len-1,0] ) # invert limits of y-axis!
    # fig.tight_layout() 
    
    def init_tracks():
        raw_img.set_data(rawframes_gam[0,:,:])
        
        for tplot in trajplots:
            tplot.set_data([],[])
        
        # heading.set_text("")
        # ax.set(title='frame: {}, time: {}s'.format(frm_start,time_start))
        # return raw_img, trajplots #,heading
    
    def update_frame(frm): #,trajdata): #,trajplots):
        raw_img.set_data(rawframes_gam[frm,:,:])
        
        # update all trajectory plots individually
        for traj,tplot in zip(trajdata.groupby('particle'),trajplots): 
            # get DataFrame of an individual particle
            _,trajdf = traj # omit the particle ID
            
            # check if current frame adds a trajectory point
            if frm in trajdf.index: 
                tplot.set_data(trajdf.loc[:frm+1].x, trajdf.loc[:frm+1].y)
        
        time = 1000*frm/fps # ms
        # heading.set_text("frame: {}\ntime: {:.1f} ms".format(frm,time))
        ax.set_title("frame: {}, time: {:.1f} ms".format(frm,time))
        
        return raw_img, trajplots#, heading
    
    traj_ani = FuncAnimation(fig, update_frame, init_func=init_tracks, 
                             frames=amnt_f, #fargs=(trajdata),#, trajplots),
                             interval=70, blit=False, repeat=False)
    return traj_ani

# anim2 = AnimateTracksOnRawData(t2_long,rawframes_ROI,settings)
# anim2.save('Au50_raw+tracks_1000frames.mp4')


def NormalizedMass(sizes_df_lin):
    plt.figure()
    
    diameter = sizes_df_lin["diameter"]
    rawmass = sizes_df_lin["rawmass"]
    
    scat_cross = rawmass / (diameter**6)
    
    plt.scatter(diameter, scat_cross)
    plt.xlabel("Diameter [nm]")
    plt.ylabel("Rawmass / (diameter^6)")
    
    plt.figure()

    
    plt.scatter(diameter, rawmass)
    plt.xlabel("Diameter [nm]")
    plt.ylabel("Rawmass")
    
    plt.figure()
    
    num_frames = sizes_df_lin["traj length"]
    
    plt.scatter(num_frames, scat_cross)
    plt.xlabel("Number of frames")
    plt.ylabel("Rawmass / (diameter^6)")


def DiamOverMass(settings, sizes_df_lin):
    "RF"  
    # plt.figure()
    # plt.plot(sizes_df_lin["diameter"], sizes_df_lin["rawmass"], '.')

    def CalcM(I_scat, I_in, d):
        m = np.sqrt(I_scat / I_in * (1/d**6))
        return m
        

    plt.figure()

    for index, row in sizes_df_lin.iterrows():
        D = row["diffusion"]
        traj = int(row["traj length"])

        CI68_low, I_mean_mean, CI68_high = nd.Simulation.RandomWalkCrossSection(settings, D, traj, num_particles = 100)

        d = row["diameter"]
        I_scat = row["rawmass"]
        
        m_low = CalcM(I_scat, CI68_low, d)
        m_mean = CalcM(I_scat, I_mean_mean, d)
        m_high = CalcM(I_scat, CI68_high, d)
        
        m_mean = np.abs(m_low + m_high) / 2
        m_error = np.abs(m_low - m_high) / 2
        
        plt.errorbar(d, m_mean, yerr = m_error)

        plt.xlabel("Diameter")
        plt.ylabel("m-factor")

def ErrorIMode():
    "RF"  
    from scipy.stats import norm
    
    r_max = 15
    r_step = 100
    D = 5
    t = 1
    sigma = np.sqrt(2*D*t)
    
    r = np.linspace(-2*r_max, 2*r_max, r_step)

    
    #Probabilty time 0
    P_r_0 = r.copy()
    P_r_0[r<0] = 0
    P_r_0[r>r_max] = 0
    
    P_r_0 = P_r_0 / np.sum(P_r_0)
    
    P_dr = norm.pdf(r, scale = sigma)
    
    P_r = np.convolve(P_r_0, P_dr, mode = "same")
    
    plt.figure()
    plt.plot(r, P_r_0, ':.')
    
    plt.plot(r, P_r, ':.')
    
   
def BayesianGaussianMixture(settings, diameter, x_min = None, x_max = None, bin_nm = None, num_comp = 1, DoPlot = True, warm_start = False, title = "No Titel given"):
    from sklearn.mixture import GaussianMixture
    import scipy.stats as stats
    from sklearn import mixture

    my_func = np.expand_dims(diameter, axis = 1)

    # my_mean_prior = np.array([50, 100, 130, np.max(diameter)])
    # my_covariance_prior = list(np.tile(10,num_comp))
    
    # my_mean_prior = np.array([np.mean(diameter)])
    # my_covariance_prior = np.array(3E0)

    # g = mixture.BayesianGaussianMixture(n_components = num_comp, covariance_type = 'spherical', max_iter = 100000, mean_precision_prior = 1E-10, tol=1E-10, mean_prior = my_mean_prior, covariance_prior = my_covariance_prior)
    
    my_means_init = np.linspace(np.min(diameter), np.max(diameter), num_comp)
    my_means_init = np.expand_dims(my_means_init, axis = 1)
    
    g = mixture.GaussianMixture(n_components = num_comp, covariance_type = 'spherical', max_iter = 100000, tol=1E-10, warm_start = warm_start, means_init = my_means_init)
    
    g.fit(my_func)
    
    weights = g.weights_
    mean = g.means_.squeeze()
    std = np.sqrt(g.covariances_.squeeze())
    
    if DoPlot == True:
        result = np.round(np.stack((weights*100, mean, std)),2)
        
        #only use weights above 0.1%
        real_component = result[0,:] > 0.1
        result = result[:,real_component]
        
        #sort by mean diameter
        result = result[:,result[1,:].argsort()]
    
        fig, axs =plt.subplots(2,1)
        
        if x_min == None:
            x_min = settings["Plot"]["Histogramm_min"]
        if x_max == None:
            x_max = settings["Plot"]["Histogramm_max"]
        if bin_nm == None:
            num_bins = settings["Plot"]["Histogramm_Bins"]
        else:
            diameter_range = np.max(diameter) - np.min(diameter)
            num_bins = int(np.round(diameter_range/bin_nm))
        
        axs[0].set_title(title)
        axs[0].hist(diameter, bins = num_bins, density = True, histtype = "step")
        axs[0].set_xlabel("Diameter [nm]")    
        axs[0].set_ylabel("Probabilty")    
        
        x = np.linspace(x_min, x_max, int(10*x_max))
        
        if num_comp == 1:
            axs[0].plot(x, stats.norm.pdf(x, mean[0][0], std))
        else:
            for ii in range(num_comp):
                if std[ii] > 0.5:
                    axs[0].plot(x, weights[ii] * stats.norm.pdf(x, mean[ii], std[ii]), "-")
                else:
                    axs[0].plot(x, weights[ii] * stats.norm.pdf(x, mean[ii], 5), ":")
        
       
        # make table with the info
        num_comp_final = result.shape[1]
        columns = ["weight in %", "mean [nm]", "std [nm]"]
        rows = ['comp: %d' % (x+1) for x in range(num_comp_final)]
        
        axs[1].axis('tight')
        axs[1].axis('off')
        
        the_table = axs[1].table(cellText=np.transpose(result),
                          rowLabels=rows,
                          colLabels=columns,
                          loc='center')

    
    return result
   

    
    
def GuidedMixGaussian(sizes_df_lin, min_traj_length = 500, num_comp = 2, central_diameter = None):
    # min_traj_length = 500
    if central_diameter == None:
        central_diameter = np.mean(sizes_df_lin.diameter)
    

    #min trajectory length
    min_length = sizes_df_lin["diameter"] / central_diameter * min_traj_length
    
    sizes_df_lin = sizes_df_lin[sizes_df_lin["traj length"] > min_length]
    # sizes_df_lin_loop = sizes_df_lin_loop[sizes_df_lin_loop.diameter < 100]
    
    use_diameter = sizes_df_lin.diameter.values

    GaussMixComp = nd.sandbox.BayesianGaussianMixture(None, use_diameter, num_comp = num_comp, x_min = 0, x_max = 300, bin_nm = 5, title = "GuidedMixGaussian")
        
    sizes_df_lin = sizes_df_lin.reset_index(drop = True)

    if "comp" not in sizes_df_lin.keys():
        sizes_df_lin["comp"] = -1
    
    if "comp_prob" not in sizes_df_lin.keys():
        sizes_df_lin["comp_prob"] = -1
    
        

    prob_comp = np.zeros(num_comp)

    for index, data in sizes_df_lin.iterrows():
        #std of the diameter
        diam_std = data['diffusion std'] / data['diffusion'] * data['diameter']
        
        diam_min = data['diameter'] + diam_std
        diam_max = data['diameter'] - diam_std
        
        # loop throug components
        for ii_comp in range(num_comp):
            comp_mean = GaussMixComp[1, ii_comp]
            comp_std = GaussMixComp[2, ii_comp]
            
            prob_comp[ii_comp] = norm.cdf(x = diam_min, loc = comp_mean, scale = comp_std) - \
                norm.cdf(x = diam_max, loc = comp_mean, scale = comp_std)
                
        # normalize
        prob_comp = prob_comp / np.sum(prob_comp)
        
        comp = np.argmax(prob_comp)
        
        # component with highest prob
        sizes_df_lin.loc[index, "comp"] = comp
        
        # probabilty
        sizes_df_lin.loc[index, "comp_prob"] = prob_comp[comp]
        
    
    
    from scipy.stats import norm
   
# In[]
# nd.Simulation.RandomWalkCrossSection(D = 13, traj_length=2000, dt=1/700, r_max = 8, ShowTraj = True, num_particles = 10, ShowReflection = True)

# nd.Simulation.RandomWalkCrossSection(D = 4, traj_length=8000, dt=1/500, r_max = 8, ShowTraj = True, num_particles = 1, ShowReflection = True)





