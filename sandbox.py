# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:45:19 2020

@author: foersterronny
"""
import numpy as np # library for array-manipulation
import matplotlib.pyplot as plt # libraries for plotting
from matplotlib.gridspec import GridSpec
from pdb import set_trace as bp #debugger


# In[]


#here is the file for playing with new functions, debugging and so on


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