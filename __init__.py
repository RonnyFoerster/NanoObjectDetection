# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 09:34:54 2019

@author: foersterronny


"""
import logging
logger = logging.getLogger(__name__)


from . import AdjustSettings
from . import CalcDiameter
from . import CheckSystem
from . import default_json
from . import Drift
from . import Experiment
from . import get_trajectorie
from . import gui
from . import handle_data
from . import JupyterLab
from . import mpl_style
from . import ParameterEstimation
from . import particleStats
from . import PlotProperties
from . import PreProcessing
from . import sandbox
from . import Simulation
from . import Start
from . import statistics
from . import teaching
from . import Theory
from . import Tools
from . import Tutorial
from . import wlsice
from . import visualize


# set up the logger for the entire module
import logging
logger = logging.getLogger("nd")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter(" %(name)s: %(levelname)s: %(module)s.%(funcName)s: %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
        
