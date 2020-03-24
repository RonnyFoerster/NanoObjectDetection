# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:45:19 2020

@author: foersterronny
"""

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

