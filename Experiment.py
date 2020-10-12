# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:49:05 2020

@author: foersterronny
"""
import numpy as np


def Dilution(d_channel, d_goal, mode = "c_stock", c_stock = None):
    # d_channel diameter of the channel in m
    # d_goal dinstance between two NP inside the fiber m
    # mode: parameters given
    #   "c_stock", NP/ml
    #   ""
    # c_stock NP/ml
    
    #channel crosssection
    A_channel = np.pi * d_channel**2 / 4
    
    #1ml = 10^-6 m^3
    
    # concentration in m^3
    if mode == 'c_stock'
        c_stock_m3 = c_stock * 1E6
    else:
        print("any other mode not implemented yet")
    
    # concentration per m fiber
    c_stock_length = c_stock_m3 * A_channel
    
    # c in NP/m
    c_goal = 1/d_goal
    
    dilution = c_stock_length / c_goal
    
    return dilution