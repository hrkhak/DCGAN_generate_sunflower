
#import modules
import sys
sys.path.insert(0,'./Desktop/DCGAN_generate_sunflower/')

from DCGAN_generate_sunflower import DataAugmentation
from DCGAN_generate_sunflower import DataPreparation
from DCGAN_generate_sunflower import Discriminator
from DCGAN_generate_sunflower import Generator
from DCGAN_generate_sunflower import Traning


#import librarys and requirements
##############################################################
import tensorflow as tf
from tensorflow.keras import layers

from scipy import ndarray

import skimage as sk
from skimage import io
from skimage import util
from skimage import transform

#%matplotlib inline

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
##########################################################################


BATCH_SIZE = 128 
LR_D = 0.0002 
LR_G = 0.0002 

BETA1 = 0.9 
EPSILON = 0.001 
LEAK_RELU_APLPHA = 0.2 

KERNEL_INITIALIZER='glorot_uniform'

######################################################################################################
#CALL GENERATOR

IMAGE_SIZE = 256
EPOCHS = 10
NOISE_SIZE = 100
NUM_NEW_IMAGES = 1000


generator = make_generator_model()

# به کار بردن جنراتور برای ایجاد یک تصویر
noise = tf.random.normal([1, NOISE_SIZE])
generated_image = generator(noise, training=False)

generated_image2 = generated_image[0].numpy() * 127.5 + 127.5

#######################################################################################################

#CALL DISCRIMINATOR

discriminator = make_discriminator_model()
#استفاده از جداساز برای کلاس بندی تصاویر فیک و اصلی 
#خروجی مثبت مدل برای تصاویر اصلی است و یرای تصاویر تقلبی خروجی منفی است
decision = discriminator(generated_image)
print (decision)

#######################################################################################################
DATASET_FOLDER = './flower_photos'

GET_DATASET = True
data_dir = './content/'
if GET_DATASET:
    print('Download and fechting DataSet')
    !wget http://download.tensorflow.org/example_images/flower_photos.tgz
    ZIP_FILE = '/content/flower_photos.tgz'
    tarfile.open(ZIP_FILE, 'r:gz').extractall(data_dir)
num_of_images = len(os.listdir(DATASET_FOLDER))

#######################################################################################################
# ایجاد دیکشنری از تبدیلات تعریف شده
available_transformations = {
    'rotate': random_rotation,
    'vertical_flip': vertical_flip,
    'horizontal_flip': horizontal_flip,
    'vertical_and_horizontal_flip': vertical_and_horizontal_flip,
    'TF_crop_pad': TF_crop_pad
}

folder_path = DATASET_FOLDER

# یافتن مسیر تمامی فایل ها
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 0
while num_generated_files <= NUM_NEW_IMAGES:
    image_path = random.choice(images)
    # خواندن تصاویر به عنوان ارایه دو بعدی از تصاویر
    image_to_transform = sk.io.imread(image_path)
    # اعمال تصادفی تعدادی از تبدیلات
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # اعمال تصادفی تبدیلات به یک تصویر
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1

        new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)

        # ذخیره تصاویر
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
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

sample_images = input_images[:5]
show_samples(sample_images)
#######################################################################################################

#######################################################################################################

#######################################################################################################
### LOSS AND OPTIMIZER
# محاسبه cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#Discriminator loss
#این روش مشخص می‌کند که تمایزکننده چقدر می‌تواند تصاویر واقعی را از تقلبی تشخیص دهد. پیش‌بینی‌های تشخیص‌دهنده روی تصاویر واقعی را با آرایه‌ای از 1s و پیش‌بینی‌های تشخیص‌دهنده در تصاویر جعلی (تولید شده) را با آرایه‌ای از 0 مقایسه می‌کند.
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
  
  
#Generator loss
#این روش مشخص می‌کند که جداکننده چقدر می‌تواند تصاویر واقعی را از فیک تشخیص دهد
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
#لازم به ذکر است که بهینه ساز جنراتور و جداساز متفاوت است به دلیل اینکه این دو دو شبکه متفاوت هستند

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_D, beta_1=BETA1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_G, beta_1=BETA1)


train(train_dataset, EPOCHS)

checkpoint.save(file_prefix = checkpoint_prefix)
RESTORE_CHECKPOINT = True
if RESTORE_CHECKPOINT:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
noise = tf.random.normal([2, NOISE_SIZE])
generated_image = generator(noise, training=False)
show_samples2(generated_image)


noise = tf.random.normal([5, NOISE_SIZE])
generated_image = generator(noise, training=False)
show_samples2(generated_image)

