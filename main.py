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













### LOSS AND OPTIMIZER
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#Discriminator loss
#This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
  
  
#Generator loss
#The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, we will compare the discriminators decisions on the generated images to an array of 1s.

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
#The discriminator and the generator optimizers are different since we will train two networks separately.

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_D, beta_1=BETA1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_G, beta_1=BETA1)
