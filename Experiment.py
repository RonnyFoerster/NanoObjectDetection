# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:49:05 2020

@author: foersterronny
"""
import numpy as np


def Dilution(c_stock, d_channel, d_goal):
    # c_stock NP/ml
    # d_channel diameter of the channel in m
    # d_goal dinstance between two NP inside the fiber m/NP fiber length
    
    #channel crosssection
    A_channel = np.pi * d_channel**2 / 4
    
    #1ml = 10^-6 m^3
    
    # concentration in m^3
    c_stock_m3 = c_stock * 1E6
    
    # concentration per m fiber
    c_stock_length = c_stock_m3 * A_channel
    
    # c in NP/m
    c_goal = 1/d_goal
    
    dilution = c_stock_length / c_goal
    
    return dilution