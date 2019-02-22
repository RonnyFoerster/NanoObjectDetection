# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:25:08 2019

@author: foersterronny
"""

import matplotlib as mpl # I import this entirely here, as it's needed to change colormaps

#def Main():
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