# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:10:52 2019

@author: foersterronny
"""

import matplotlib as mpl # I import this entirely here, as it's needed to change colormaps

params = {
   'figure.figsize': [8, 6],
   'legend.fontsize': 12,
   'text.usetex': False,

   'ytick.labelsize': 10,
   'ytick.direction': 'in',
   'xtick.labelsize': 10,
   'xtick.direction': 'in',
   'font.size': 10,
   }
mpl.rcParams.update(params)

title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
  'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'12'}



def export(save_folder_name, save_image_name):
    
    import matplotlib.pyplot as plt # Libraries for plotting
    import datetime
    import os

    plt.show()
    
    my_dir_name = '%s\\{date:%y%m%d}\\'.format( date=datetime.datetime.now()) %save_folder_name
    
    try:
        os.stat(my_dir_name)
    except:
        os.mkdir(my_dir_name) 
    
    file_name = '{date:%H%M%S}_%s.png'.format( date=datetime.datetime.now()) %save_image_name
    plt.savefig(my_dir_name +  file_name, dpi=400, bbox_inches='tight')
    
    print('Figure saved')