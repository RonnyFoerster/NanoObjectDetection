# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 08:54:11 2019

@author: foersterronny
"""


import numpy as np
import logging

import NanoObjectDetection as nd

from packaging import version
import trackpy as tp




def LoggerSetLevel(level, TryLogger = False):
    """
    set the minimum level when a logging message is plotted    

    Parameters
    ----------
    level : TYPE
        options: "debug", "info", "warning", "error", "critical".
    TryLogger : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
       
    if level == "debug":
        nd.logger.setLevel(logging.DEBUG)
    elif level == "info":
        nd.logger.setLevel(logging.INFO)
    elif level == "warning":
        nd.logger.setLevel(logging.WARNING)
    elif level == "error":
        nd.logger.setLevel(logging.ERROR)
    elif level == "critical":
        nd.logger.setLevel(logging.CRITICAL)
    else:
        nd.logger.error("Level unknown. Choose debug, info, warning, error or critical in the json file.")

    if TryLogger == True:
        nd.logger.debug("TEST LOGGER MODE: <debug> ON")
        nd.logger.info("TEST LOGGER MODE: <info> ON")
        nd.logger.warning("TEST LOGGER MODE: <warning> ON")
        nd.logger.error("TEST LOGGER MODE: <error> ON")
        nd.logger.critical("TEST LOGGER MODE: <critical> ON")




def GetTpVersion():
    """
    return the main version of trackpy

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    tp_4 = '0.4'
    tp_5 = '0.5'
    tp_version = version.parse(tp.__version__)
    
    
    if ((tp_version >= version.parse(tp_4)) & (tp_version < version.parse(tp_5))):
        tp_version = 4
        
    elif tp_version >= version.parse(tp_5):
        tp_version = 5
        
    # nd.logger.debug("Trackpy main version: %s", 5)
    
    return tp_version


