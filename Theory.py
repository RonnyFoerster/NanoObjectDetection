# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 08:44:12 2020

@author: foersterronny
"""

# HERE ARE ALL STANDARD EQUATIONS IN PHYSICS LIKE CONVERSIONS ETC.

def PulseEnergy2CW(E_pulse, rep_rate):
    """
    E_pulse: pulse power [W]
    rep_rate: Repeatition rate [Hz]
    
    https://www.thorlabs.com/images/tabimages/Laser_Pulses_Power_Energy_Equations.pdf
    """
    
    # average power
    P_avg = E_pulse * rep_rate
    
    return P_avg
    
    
    