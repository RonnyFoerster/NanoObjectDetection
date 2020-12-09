# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:49:05 2020

@author: foersterronny
"""
import numpy as np



def Dilution(d_channel, d_goal, mode = "c_stock", c_stock = None):
    """
    d_channel: diameter of the channel in m
    d_goal: distance between two NP inside the fiber m
    mode: parameters given
        "c_stock", NP/ml
    
    c_stock: NP/ml
    """
    
    #channel crosssection
    A_channel = np.pi * d_channel**2 / 4
    
    #1ml = 10^-6 m^3
    
    # concentration in m^3
    if mode == 'c_stock':
        c_stock_m3 = c_stock * 1E6
    else:
        print("any other mode not implemented yet")
    
    # concentration per m fiber
    c_stock_length = c_stock_m3 * A_channel
    
    # c in NP/m
    c_goal = 1/d_goal
    
    dilution = c_stock_length / c_goal
    
    return dilution



def GetViscosity(temperature = 295.15, solvent = "water"):
    """ import viscosity value from CoolProp data base or ask for user input   

    Parameters
    ----------
    temperature : FLOAT, optional
        in Kelvin. The default is 295.15.
    solvent : STR, optional
        name of the liquid that surrounds the particles. The default is "water".

    Returns
    -------
    my_visc : FLOAT
        viscosity value in Ns/um^2, i.e. ready to be saved in the parameter.json file

    """
    try:
        import CoolProp as CP
        
        my_visc = CP.CoolProp.PropsSI('V','T',temperature,'P',101.3,solvent)
        print(r"\nCoolProp (http://www.coolprop.org/) returns a viscosity of {:.3e} Ns/m^2.".format(my_visc))
        # convert from Ns/m^2 (=Pa*s) to Ns/um^2
        my_visc = my_visc * 1e-12
        
    except ModuleNotFoundError:
        print("CoolProp is not installed yet. Please enter the correct viscosity yourself.")
        my_visc = 1e-12 * float(input("viscosity [Ns/m^2] = "))
		
		# https://www.tec-science.com/de/mechanik/gase-und-fluessigkeiten/viskositat-von-flussigkeiten-und-gasen/

    return my_visc

