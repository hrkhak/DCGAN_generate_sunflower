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
import random
import time
import tarfile
tz_NY = pytz.timezone('America/New_York') 
##########################################################################3
IMAGE_SIZE = 256
EPOCHS = 10
NOISE_SIZE = 100
NUM_NEW_IMAGES = 1000
DATASET_FOLDER = 'only_flowers/'

BATCH_SIZE = 128 # Paper
LR_D = 0.0002 # Paper
LR_G = 0.0002 # Paper

BETA1 = 0.9 # Default
EPSILON = 0.001 # Default
LEAK_RELU_APLPHA = 0.2 # Paper

KERNEL_INITIALIZER='glorot_uniform' # Default

######################################################################################################
#CALL GENERATOR
generator = make_generator_model()

# Use the (as yet untrained) generator to create an image.
noise = tf.random.normal([1, NOISE_SIZE])
generated_image = generator(noise, training=False)

generated_image2 = generated_image[0].numpy() * 127.5 + 127.5

#######################################################################################################

#CALL DISCRIMINATOR

discriminator = make_discriminator_model()
#Use the (as yet untrained) discriminator to classify the generated images as real or fake. 
#The model will be trained to output positive values for real images, and negative values for fake images.
decision = discriminator(generated_image)
print (decision)

#######################################################################################################


GET_DATASET = True
data_dir = '/content/flower_photos'
#DATASET_FOLDER = 'content/flower_photos/'
if GET_DATASET:
    print('Fechting the Image Data Set')
    !wget http://download.tensorflow.org/example_images/flower_photos.tgz
    print('Unzipping the file')
    ZIP_FILE = '/content/flower_photos.tgz'
    tarfile.open(ZIP_FILE, 'r:gz').extractall(data_dir)
    print('Daisies images are available')
#Check if there are some files in the folder
num_of_images = len(os.listdir(DATASET_FOLDER))

#######################################################################################################
# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'vertical_flip': vertical_flip,
    'horizontal_flip': horizontal_flip,
    'vertical_and_horizontal_flip': vertical_and_horizontal_flip,
    'TF_crop_pad': TF_crop_pad
}

folder_path = DATASET_FOLDER

# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 0
while num_generated_files <= NUM_NEW_IMAGES:
    # random image from the folder
    image_path = random.choice(images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    # random num of transformation to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1

        new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)

        # write image to the disk
        #io.imsave(new_file_path, transformed_image.astype(np.uint8))
        io.imsave(new_file_path, transformed_image)
        num_generated_files += 1


#######################################################################################################
input_images = np.asarray([np.asarray(
    Image.open(file)
    .resize((IMAGE_SIZE, IMAGE_SIZE))
    ) for file in glob(DATASET_FOLDER+'*')])
print ("Input: " + str(input_images.shape))

np.random.shuffle(input_images)

train_images = input_images.reshape(input_images.shape[0], 256, 256, 3).astype('float32')
train_images = (train_images - 127.5) / 127.5 

BUFFER_SIZE = input_images.shape[0]
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

sample_images = input_images[:5]
show_samples(sample_images)
#######################################################################################################

#######################################################################################################

#######################################################################################################
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


train(train_dataset, EPOCHS)

checkpoint.save(file_prefix = checkpoint_prefix)
# upload the files (checkpoint, ckpt-xxx.index, ckpt-xxx.data-*) into training_checkpoints folder
RESTORE_CHECKPOINT = True
if RESTORE_CHECKPOINT:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
noise = tf.random.normal([2, NOISE_SIZE])
generated_image = generator(noise, training=False)
show_samples2(generated_image)


noise = tf.random.normal([5, NOISE_SIZE])
generated_image = generator(noise, training=False)
show_samples2(generated_image)







