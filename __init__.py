# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 09:34:54 2019

@author: foersterronny


"""
# import logging
# logger = logging.getLogger(__name__)


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


# In[logger]  set up the logger for the entire module
import logging
try:
    del logger
except:
    pass


# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    # https://tforgione.fr/posts/ansi-escape-codes/
    # font color "\x1B[38;2;R;G;Bm" with RGB numbers from 0-255

    blue   = "\x1B[38;2;0;200;255m"
    green  = "\x1B[38;2;0;255;0m"
    yellow = "\x1B[38;2;255;255;0m"
    orange = "\x1B[38;2;255;128;0m"
    red    = "\x1B[38;2;255;50;0m"
    reset  = "\x1b[0m"
    format = "%(name)s-%(levelname)-8s: %(message)s (%(filename)s: %(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: orange + format + reset,
        logging.CRITICAL: red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("nd")
logger.propagate = False


stream_handler = logging.StreamHandler()
stream_handler.setFormatter(CustomFormatter())
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

# logger.debug("debug")
# logger.info("info")
# logger.warning("warning")

#lets run the logger
Tools.LoggerSetLevel("info", TryLogger = False)