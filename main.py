#import librarys and requirements
##############################################################
import tensorflow as tf
from tensorflow.keras import layers

from scipy import ndarray

import skimage as sk
from skimage import io
from skimage import util
from skimage import transform

%matplotlib inline

from datetime import datetime
import os
from glob import glob
from IPython import display
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import pytz
tz_NY = pytz.timezone('America/New_York') 
import random
import time
