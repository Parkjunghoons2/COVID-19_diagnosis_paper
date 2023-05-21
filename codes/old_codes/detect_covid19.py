 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:20:21 2021

@author: nextgen
"""

import os

import configs

import tensorflow as tf

import re

import glob

import tarfile

import os
import warnings
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import random
from PIL import Image
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageOps
import math
import os
import random
import copy
import scipy
import imageio
import string
import numpy as np

from os import listdir
from os.path import isfile, join
mypath = '/home/nextgen/Desktop/data/pjh/images/images/images'
targetdir= '/home/nextgen/Desktop/data/pjh/images/images/images'
fileExt = r'.png'
onlyfiles = [os.path.join(mypath, _) for _ in os.listdir(mypath) if _.endswith(fileExt)]
onlyfiles1 = onlyfiles[0:-1000]
onlyfiles2 = onlyfiles[-1001:-1]
len(onlyfiles)

from skimage.transform import resize

try:  # SciPy >= 0.19
    from scipy.special import comb

except ImportError:

    from scipy.misc import comb


def bernstein_poly(i, n, t):

    """

     The Bernstein polynomial of n, i as a function of t

    """

 

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

 

def bezier_curve(points, nTimes=1000):

    """

       Given a set of control points, return the

       bezier curve defined by the control points.

 

       Control points should be a list of lists, or list of tuples

       such as [ [1,1], 

                 [2,3], 

                 [4,5], ..[Xn, Yn] ]

        nTimes is the number of time steps, defaults to 1000

 

        See http://processingjs.nihongoresources.com/bezierinfo/

    """

 

    nPoints = len(points)

    xPoints = np.array([p[0] for p in points])

    yPoints = np.array([p[1] for p in points])

 

    t = np.linspace(0.0, 1.0, nTimes)

 

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    

    xvals = np.dot(xPoints, polynomial_array)

    yvals = np.dot(yPoints, polynomial_array)

 

    return xvals, yvals

 

def data_augmentation(x, y, prob=0.5):

    # augmentation by flipping

    cnt = 3

    while random.random() < prob and cnt > 0:

        degree = random.choice([0, 1, 2])

        x = np.flip(x, axis=degree)

        y = np.flip(y, axis=degree)

        cnt = cnt - 1

 

    return x, y

 

def nonlinear_transformation(x, prob=0.5):

    if random.random() >= prob:

        return x

    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]

    xpoints = [p[0] for p in points]

    ypoints = [p[1] for p in points]

    xvals, yvals = bezier_curve(points, nTimes=100000)

    if random.random() < 0.5:

        # Half change to get flip

        xvals = np.sort(xvals)

    else:

        xvals, yvals = np.sort(xvals), np.sort(yvals)

    nonlinear_x = np.interp(x, xvals, yvals)

    return nonlinear_x

 

def local_pixel_shuffling(x, prob=0.5):

    if random.random() >= prob:

        return x

    image_temp = copy.deepcopy(x)

    orig_image = copy.deepcopy(x)

    img_rows, img_cols, img_deps = x.shape

    num_block = 500

    for _ in range(num_block):

        block_noise_size_x = random.randint(1, img_rows//10)

        block_noise_size_y = random.randint(1, img_cols//10)

        #block_noise_size_z = random.randint(0, 3)

        noise_x = random.randint(0, img_rows-block_noise_size_x)

        noise_y = random.randint(0, img_cols-block_noise_size_y)

        #noise_z = random.randint(0, img_deps-block_noise_size_z)

        if img_deps >3 :

            filters = 3

            window = orig_image[noise_x:noise_x+block_noise_size_x, 

                               noise_y:noise_y+block_noise_size_y, 

                               #noise_z:noise_z+block_noise_size_z,
                                :,
                           ]

            window = window.flatten()

            np.random.shuffle(window)

            window = window.reshape((block_noise_size_x, 

                                 block_noise_size_y, 
                                 img_deps))
                                 #block_noise_size_z))

            image_temp[noise_x:noise_x+block_noise_size_x, 

                      noise_y:noise_y+block_noise_size_y, 
                      :] = window
                      #noise_z:noise_z+block_noise_size_z] = window

        else :

            window = orig_image[noise_x:noise_x+block_noise_size_x, 

                               noise_y:noise_y+block_noise_size_y, 

                               #noise_z:noise_z+block_noise_size_z,
                                :,
                           ]

            window = window.flatten()

            np.random.shuffle(window)

            window = window.reshape((block_noise_size_x, 

                                 block_noise_size_y, 
                                 img_deps))

                                 #block_noise_size_z))

            image_temp[noise_x:noise_x+block_noise_size_x, 

                      noise_y:noise_y+block_noise_size_y, 
                       :] = window

                      #noise_z:noise_z+block_noise_size_z] = window

    local_shuffling_x = image_temp

 

    return local_shuffling_x

 

def image_in_painting(x):

    img_rows, img_cols, img_deps = x.shape

    cnt = 5

    while cnt > 0 and random.random() < 0.95:

        block_noise_size_x = random.randint(img_rows//6, img_rows//3)

        block_noise_size_y = random.randint(img_cols//6, img_cols//3)

        block_noise_size_z = random.randint(0,3)

        noise_x = random.randint(3, img_rows-block_noise_size_x-3)

        noise_y = random.randint(3, img_cols-block_noise_size_y-3)

        noise_z = random.randint(0, img_deps-block_noise_size_z)
        
        x_point = random.randint(noise_x, noise_x+block_noise_size_x)
        y_point = random.randint(noise_y, noise_y+block_noise_size_y)
        
        inpainting = np.zeros((block_noise_size_x,block_noise_size_y,3))
        
        for row in range(0,inpainting.shape[0]-1):
            for col in range(0,inpainting.shape[1]-1):
                inpainting[row,col] = x[x_point,y_point]

        x[

          noise_x:noise_x+block_noise_size_x, 

          noise_y:noise_y+block_noise_size_y, 

          :] = inpainting *1.0

        cnt -= 1

    return x



 

def image_out_painting(x):

    img_rows, img_cols, img_deps = x.shape

    image_temp = copy.deepcopy(x)
    

    x = np.ones((img_rows, img_cols, img_deps) ) * 1.0
    
    x_point = random.randint(0, img_rows-1)
    y_point = random.randint(0, img_cols-1)
    
    for row in range(0,x.shape[0]):
        for col in range(0,x.shape[1]):
            x[row,col] = image_temp[x_point,y_point]
    
    cnt = 4
    while cnt>0:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)

        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)

        block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)

        noise_x = random.randint(3, img_rows-block_noise_size_x-3)

        noise_y = random.randint(3, img_cols-block_noise_size_y-3)

        noise_z = random.randint(0, img_deps-block_noise_size_z)

        x[

          noise_x:noise_x+block_noise_size_x, 

          noise_y:noise_y+block_noise_size_y, 

          :] = image_temp[ noise_x:noise_x+block_noise_size_x, 

                                                       noise_y:noise_y+block_noise_size_y, 
                                                       :] * 1.0
        cnt -= 1

                                                       #noise_z:noise_z+block_noise_size_z]
    '''
    cnt = 4

    while cnt > 0 and random.random() < 0.95:

        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)

        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)

        block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)

        noise_x = random.randint(3, img_rows-block_noise_size_x-3)

        noise_y = random.randint(3, img_cols-block_noise_size_y-3)

        noise_z = random.randint(0, block_noise_size_z)
        
        x_point = random.randint(noise_x, noise_x+block_noise_size_x)
        y_point = random.randint(noise_y, noise_y+block_noise_size_y)
        
        outpainting = np.zeros((block_noise_size_x,block_noise_size_y,3))
        
        for row in range(0,outpainting.shape[0]):
            for col in range(0,outpainting.shape[1]):
                outpainting[row,col] = image_temp[x_point,y_point]

        x[

          noise_x:noise_x+block_noise_size_x, 

          noise_y:noise_y+block_noise_size_y, 

          :] = outpainting * 1.0

        cnt -= 1
    '''
    return x


 

def generate_pair(path):
    
    #path = '/home/ubuntu/Desktop/data/pjh/images/images/'
    #config = configs.models_genesis_config
    #path = r'/home/ubuntu/Desktop/data/pjh/images/images'
    #files = os.listdir(targetdir)
    #condition='*.png'
    #fileExt = r".png"
    #onlyfiles = [os.path.join(targetdir,_) for _ in os.listdir(targetdir) if _.endswith(fileExt)]
    #onlyfiles = onlyfiles[0:10]
    #for i in range(len(onlyfiles)):
    config = configs.models_genesis_config

    img = Image.open(path)

    img = img.convert('RGB')

    img = img.resize((512,512))

    #img= tf.keras.preprocessing.image.img_to_array(img)

    img = np.array(img)

    img_rows, img_cols, img_deps = img.shape[0], img.shape[1], img.shape[2]

    while True:

        y = img/255

        x = copy.deepcopy(y)            

        # Autoencoder

        x = copy.deepcopy(y)

            

        # Flip

        x, y = data_augmentation(x, y, config.flip_rate)

 

        # Local Shuffle Pixel

        x = local_pixel_shuffling(x, prob=config.local_rate)

            

        # Apply non-Linear transformation with an assigned probability

        x = nonlinear_transformation(x, config.nonlinear_rate)

            

        # Inpainting & Outpainting

        if random.random() < config.paint_rate:

            if random.random() < config.inpaint_rate:

                # Inpainting

                x = image_in_painting(x)

            else:

                # Outpainting

                x = image_out_painting(x)


        '''
        # Save sample images module

        if config.save_samples is not None and status == "train" and random.random() < 0.01:

            n_sample = random.choice( [i for i in range(conf.batch_size)] )

            sample_1 = np.concatenate((x[n_sample,0,:,2*img_deps//6].reshape(x[n].shape[0],1), y[n_sample,0,:,2*img_deps//6].reshape(x[n].shape[0],1)), axis=1)

            sample_2 = np.concatenate((x[n_sample,0,:,3*img_deps//6].reshape(x[n].shape[0],1), y[n_sample,0,:,3*img_deps//6].reshape(x[n].shape[0],1)), axis=1)

            sample_3 = np.concatenate((x[n_sample,0,:,4*img_deps//6].reshape(x[n].shape[0],1), y[n_sample,0,:,4*img_deps//6].reshape(x[n].shape[0],1)), axis=1)

            sample_4 = np.concatenate((x[n_sample,0,:,5*img_deps//6].reshape(x[n].shape[0],1), y[n_sample,0,:,5*img_deps//6].reshape(x[n].shape[0],1)), axis=1)

            final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)

            #final_sample = final_sample * 255.0

            final_sample = final_sample.astype(np.float32)

            #file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples

            #imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

         '''

        return x, y
    

def get_nih1():
    targetdir= '/home/nextgen/Desktop/data/pjh/images/images/images'
    fileExt = r'.png'
    #onlyfiles = [os.path.join(mypath, _) for _ in os.listdir(mypath) if _.endswith(fileExt)]
    onlyfiles = [os.path.join(targetdir,_) for _ in os.listdir(targetdir) if _.endswith(fileExt)]
    onlyfiles = onlyfiles[0:-1000]
    #onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    config = configs.models_genesis_config
    
    for i in range(len(onlyfiles)):

        img1,img2 = generate_pair(onlyfiles[i])
    
        yield (img1,img2)

def get_nih2():
    targetdir= '/home/nextgen/Desktop/data/pjh/images/images/images'
    files = os.listdir(targetdir)
    condition='*.png'
    fileExt = r".png"
    #onlyfiles = [os.path.join(mypath, _) for _ in os.listdir(mypath) if _.endswith(fileExt)]
    onlyfiles = [os.path.join(targetdir,_) for _ in os.listdir(targetdir) if _.endswith(fileExt)]
    onlyfiles = onlyfiles[-1001:-1]
    #onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    config = configs.models_genesis_config
    
    for i in range(len(onlyfiles)):

        img1,img2 = generate_pair(onlyfiles[i])
    
        yield (img1,img2)

train_dataset = tf.data.Dataset.from_generator(get_nih1,
                                         output_shapes=((512,512,3), (512,512,3)),
                                         output_types=(tf.float32,tf.float32))

valid_dataset = tf.data.Dataset.from_generator(get_nih2,
                                         output_shapes=((512,512,3), (512,512,3)),
                                         output_types=(tf.float32,tf.float32))                                        

train_dataset = train_dataset.shuffle(150).batch(8).repeat(6)
valid_dataset = valid_dataset.shuffle(150).batch(8).repeat(6)


import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras


def unet(pretrained_weights = None,input_size = (512,512,3)):
    inputs = tf.keras.Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)


    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'relu')(conv9)
    final = UpSampling3D(size=(1,1,3))(conv10)

    model = Model(inputs = inputs, outputs = final)

    #model.summary()
    if(pretrained_weights):

    	model.load_weights(pretrained_weights)

    return model

def unet_cbam(pretrained_weights=None, input_size=(512, 512, 3), kernel_size=3, ratio=3, activ_regularization=0.01):
    inputs = tf.keras.Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(inputs)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(conv1)

    conv1 = CBAM_attention(conv1, ratio, kernel_size, dr_ratio, activ_regularization)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(pool1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(conv2)

    conv2 = CBAM_attention(conv2, ratio, kernel_size, dr_ratio, activ_regularization)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(pool2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(conv3)

    conv3 = CBAM_attention(conv3, ratio, kernel_size, dr_ratio, activ_regularization)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(pool3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(conv4)

    conv4 = CBAM_attention(conv4, ratio, kernel_size, dr_ratio, activ_regularization)

    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(pool4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(conv5)

    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(
    UpSampling2D(size=(2, 2))(drop5))

    merge6 = concatenate([drop4, up6], axis=3)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(merge6)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(conv6)

    conv6 = CBAM_attention(conv6, ratio, kernel_size, dr_ratio, activ_regularization)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(
    UpSampling2D(size=(2, 2))(conv6))

    merge7 = concatenate([conv3, up7], axis=3)

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(merge7)

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(conv7)

    conv7 = CBAM_attention(conv7, ratio, kernel_size, dr_ratio, activ_regularization)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(
    UpSampling2D(size=(2, 2))(conv7))

    merge8 = concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(merge8)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(
    UpSampling2D(size=(2, 2))(conv8))

    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(merge9)

    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(conv9)

    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(conv9)

    conv10 = Conv2D(1, 1, activation='relu', kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))(conv9)

    final = UpSampling3D(size=(1, 1, 3))(conv10)

    model = Model(inputs=inputs, outputs=final)

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def CBAM_attention(inputs,ratio,kernel_size,dr_ratio,activ_regularization):
    x = inputs
    channel = x.get_shape()[-1]

    ##channel attention##
    avg_pool = tf.reduce_mean(x, axis=[1,2], keepdims=True)
    avg_pool = Dense(units = channel//ratio ,activation='relu', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),use_bias=True,bias_initializer='zeros',trainable=True)(avg_pool)
    avg_pool = Dense(channel, activation = 'relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True, bias_initializer='zeros',trainable=True)(avg_pool)

    max_pool = tf.reduce_max(x, axis=[1,2], keepdims=True)
    max_pool = Dense(units = channel//ratio, activation='relu', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), use_bias=True,bias_initializer='zeros',trainable=True)(max_pool)
    max_pool = Dense(channel, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), activity_regularizer=tf.keras.regularizers.l1(activ_regularization),use_bias=True, bias_initializer='zeros',trainable=True)(max_pool)
    f = Add()([avg_pool, max_pool])
    f = Activation('sigmoid')(f)

    after_channel_att = multiply([x, f])

    ##spatial attention##
    kernel_size = kernel_size
    avg_pool_2 = tf.reduce_mean(x, axis=[1,2], keepdims=True)
    max_pool_2 = tf.reduce_max(x, axis=[1,2], keepdims=True)
    concat = tf.concat([avg_pool,max_pool],3)
    concat = Conv2D(filters=1, kernel_size=[kernel_size,kernel_size], strides=[1,1], padding='same', kernel_initializer=kernel_initializer,use_bias=False)(concat)
    concat = Activation('sigmoid')(concat)
    ##final_cbam##
    attention_feature = multiply([x,concat])
    return attention_feature

ratio=8
activ_regularization=0.0001
kernel_size=7
kernel_initializer = tf.keras.initializers.VarianceScaling()
dr_ratio=0.2
targetdir = '/home/nextgen/Desktop/data/pjh/images/images'

checkpoint_path1 = './pretrained_weights_nih.ckpt'
checkpoint_dir1 = os.path.join(os.getcwd()+checkpoint_path1)
checkpoint_path2 = './pretrained_weights/base.ckpt'
checkpoint_dir2 = os.path.join(os.getcwd()+checkpoint_path2)
cp_callback1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path1,
                                                save_weights_only=True,
                                                monitor='val_loss',
                                                save_best_only=True,
                                                mode='min')
cp_callback2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path2,
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                mode='max')



def ssim_loss_minusone(y_true, y_pred):
    return 1-tf.reduce_mean(tf.image.ssim(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), 2.0))

def ssim_loss(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), 2.0))

#Model: Input Image size: 32X32X1 output Image size: 28X28X1 
#check model.summary

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    unet_cbam_model = unet_cbam(pretrained_weights = 'nih_unet_cbam_20.h5', input_size=(512,512,3),kernel_size=3, ratio=3, activ_regularization=0.01 )
    unet_cbam_model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3),
            loss='mse',
            metrics=['mse',ssim_loss])

history = unet_cbam_model.fit(train_dataset,
                            validation_data=valid_dataset, 
                            validation_steps=len(onlyfiles2)//8,
                            steps_per_epoch=len(onlyfiles1)//8, 
                            epochs=8,
                            #max_queue_size=configs.models_genesis_config.max_queue_size, 
                            #workers=configs.models_genesis_config.workers, 
                            use_multiprocessing=True, 
                            shuffle=True,
                            verbose=configs.models_genesis_config.verbose,
                            callbacks=[cp_callback1]
                           )

unet_cbam_model.save_weights('nih_unet_cbam_mse.h5')

#unet_cbam_model.load_weights('nih_unet_cbam_20_mse.h5')





#####################################################################################
#####################classification##################################################
#####################################################################################
#unet_cbam_model = unet_cbam(pretrained_weights = 'nih_unet_cbam_20.h5', input_size=(512,512,3),kernel_size=3, ratio=3, activ_regularization=0.000001 )

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    unet_cbam_model = unet_cbam(pretrained_weights = 'nih_unet_cbam_20.h5', input_size=(512,512,3),kernel_size=3, ratio=3, activ_regularization=0.000001 )

    unet_cbam_classification = Model(inputs = unet_cbam_model.input, outputs= unet_cbam_model.get_layer('dropout_1').output)

    my_model3 = tf.keras.Sequential()

    my_model3.add(unet_cbam_classification)
    
    my_model3.add(Conv2D(512,(2,2), activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

    my_model3.add(Conv2D(256,(2,2), activation = 'relu', padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

    my_model3.add(BatchNormalization())    

    #my_model2.add(cbam_tf.cbam_block(my_model2.output,name=None, ratio=2))

    my_model3.add(Conv2D(128,(2,2), activation = 'relu', padding = 'same',  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

    my_model3.add(Conv2D(64,(2,2), activation = 'relu' , padding = 'same',  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

    #my_model2.add(cbam_tf.cbam_block(my_model2.output,name=None,ratio=2))

    my_model3.add(BatchNormalization()) 

    my_model3.add(MaxPooling2D(pool_size=(2,2)))

    my_model3.add(GlobalAveragePooling2D())
    
    my_model3.add(BatchNormalization()) 
    
    my_model3.add(Flatten())

    my_model3.add(Dropout(0.3))

    my_model3.add(Dense(128,  activation = 'relu' , kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

    my_model3.add(Dropout(0.5))

    my_model3.add(Dense(64,  activation='relu' , kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

    my_model3.add(Dropout(0.3))

    my_model3.add(Dense(32, activation='relu' ,kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

    my_model3.add(Dropout(0.2))

    my_model3.add(Dense(16, activation='relu' ,kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

    my_model3.add(Dense(8, activation = 'relu' , kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

    #my_model3.add(Dropout(0.3))

    #my_model3.add(BatchNormalization())

    my_model3.add(Dense(3, activation = 'softmax'))

    my_model3.layers[0].trainable = True
    
    checkpoint_path1 = '/home/nextgen/Desktop/tf2/covid/training_1/cp7.ckpt'
    my_model3.load_weights(checkpoint_path1 )

    my_model3.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])


activ_regularization=0.00001

unet_cbam_model = unet_cbam(pretrained_weights = 'nih_unet_cbam_20.h5', input_size=(512,512,3),kernel_size=3, ratio=3, activ_regularization=0.000001 )

unet_cbam_classification = Model(inputs = unet_cbam_model.input, outputs= unet_cbam_model.get_layer('dropout_1').output)

my_model3 = tf.keras.Sequential()

my_model3.add(unet_cbam_classification)

my_model3.add(Conv2D(512,(2,2), activation = 'relu' ,padding = 'same',  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

my_model3.add(Conv2D(256,(2,2), activation = 'relu' , padding = 'same',  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

#my_model2.add(cbam_tf.cbam_block(my_model2.output,name=None, ratio=2))

my_model3.add(BatchNormalization())

my_model3.add(Conv2D(128,(2,2), activation = 'relu' , padding = 'same',  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

my_model3.add(Conv2D(64,(2,2), activation = 'relu' , padding = 'same',  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

#my_model2.add(cbam_tf.cbam_block(my_model2.output,name=None,ratio=2))

my_model3.add(BatchNormalization())

my_model3.add(MaxPooling2D(pool_size=(2,2)))

my_model3.add(GlobalAveragePooling2D())

my_model3.add(BatchNormalization())

my_model3.add(Flatten())

my_model3.add(Dropout(0.3))

my_model3.add(Dense(128, activation = 'relu', activity_regularizer=tf.keras.regularizers.l2(activ_regularization),kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

#my_model3.add(Dropout(0.5))

my_model3.add(Dense(64,  activation='relu',activity_regularizer=tf.keras.regularizers.l2(activ_regularization),  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

#my_model3.add(Dropout(0.4))

my_model3.add(Dense(32, activation='relu' ,activity_regularizer=tf.keras.regularizers.l2(activ_regularization)))

my_model3.add(Dropout(0.3))

my_model3.add(Dense(16, activation='relu' ))

my_model3.add(Dense(8, activation = 'relu'))

#my_model3.add(Dropout(0.2))

#my_model3.add(BatchNormalization())

my_model3.add(Dense(3, activation='softmax'))

my_model3.layers[0].trainable = True

my_model3.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.000001, clipvalue=2, clipnorm=1),
            loss='categorical_crossentropy',
            metrics=['accuracy'])


print('------------------loading data--------------------')

#data
xr_train_rgb = np.load('/home/nextgen/Desktop/tf2/covid/xr_train.npy')
xr_valid_rgb = np.load('/home/nextgen/Desktop/tf2/covid/xr_valid.npy')
xr_test_rgb = np.load('/home/nextgen/Desktop/tf2/covid/xr_test.npy')
labels_train_xr = np.load('/home/nextgen/Desktop/tf2/covid/labels_train_xr.npy')
labels_valid_xr = np.load('/home/nextgen/Desktop/tf2/covid/labels_valid_xr.npy')
labels_test_xr = np.load('/home/nextgen/Desktop/tf2/covid/labels_test_xr.npy')


from tensorflow.keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(rotation_range = 20,
                        zoom_range = 0.15,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.15,
                        horizontal_flip=True,
                        fill_mode='nearest')

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.8f." % (epoch, scheduled_lr))


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr

LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (50, 0.00001),
    (60, 0.00001),
    (70, 0.00001),
    (80, 0.000008),
    (130, 0.000005),
    (150, 0.000004),
    (200, 0.000003),
    (250, 0.000003),
    (280, 0.000002),
    (300, 0.000005),
    (350, 0.000003),
    (400, 0.000002),
    (450, 0.000002),
    (700, 0.000001)
]


checkpoint_path1 = '/home/nextgen/Desktop/tf2/covid/training_1/cp2.ckpt'
my_model3.load_weights(checkpoint_path1 )


from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=8, verbose=1, mode='min', min_lr=5e-7)


print('------------------fitting----------------------------')
my_history = my_model3.fit(aug.flow(xr_train_rgb, np.array(labels_train_xr), batch_size=4),
	validation_data=(xr_valid_rgb,np.array(labels_valid_xr)),
    steps_per_epoch=xr_train_rgb.shape[0]//4, # number of images comprising of one epoch
    validation_steps=xr_valid_rgb.shape[0]//4,
    callbacks=[cp_callback1,cp_callback2,lr_reduce],#CustomLearningRateScheduler(lr_schedule)],
	epochs=120)



print('------------------saving weights------------------------')
my_model3.save_weights('ssim_classification2.h5')




from tensorflow.keras import optimizers
learning_rate = 0.0001

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    unet_cbam_model = unet_cbam(pretrained_weights = 'nih_unet_cbam_20.h5', input_size=(512,512,3),kernel_size=3, ratio=3, activ_regularization=0.000001 )

    unet_cbam_classification = Model(inputs = unet_cbam_model.input, outputs= unet_cbam_model.get_layer('dropout_1').output)

    x = unet_cbam_classification.output
    x = Flatten()(x) # Flatten dimensions to for use in FC layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x = Dense(3, activation='softmax')(x) # Softmax for multiclass
    transfer_model = Model(inputs=unet_cbam_classification.input, outputs=x)
    transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

history1 = transfer_model.fit(aug.flow(xr_train_rgb, np.array(labels_train_xr), batch_size=32),
	validation_data = (xr_valid_rgb,np.array(labels_valid_xr)),
    steps_per_epoch=xr_train_rgb.shape[0]//32, # number of images comprising of one epoch
    validation_steps=xr_valid_rgb.shape[0]//32,
    callbacks=[cp_callback1,cp_callback2,CustomLearningRateScheduler(lr_schedule)],
	epochs=50)



#my_model3.load_weights('ssim_classification3.h5')
checkpoint_path1 = '/home/nextgen/Desktop/tf2/covid/training_1/cp.ckpt'
checkpoint_path2 = '/home/nextgen/Desktop/tf2/covid/pretrained_weights/cp.ckpt'

checkpoint_path3 = './training_1/cp2.ckpt'
checkpoint_path4 = './pretrained_weights/cp2.ckpt'

checkpoint_path5 = './training_1/cp3.ckpt'
checkpoint_path6 = './pretrained_weights/cp3.ckpt'

checkpoint_path7 = './training_1/cp4.ckpt'
checkpoint_path8 = './pretrained_weights/cp4.ckpt'

checkpoint_path9 = '/home/nextgen/Desktop/tf2/covid/training_1/cp5.ckpt'
checkpoint_path10 = '/home/nextgen/Desktop/tf2/covid/training_1//cp5.ckpt'

checkpoint_path11 = '/home/nextgen/Desktop/tf2/covid/training_1/cp6.ckpt'
checkpoint_path12 = '/home/nextgen/Desktop/tf2/covid/training_1/cp6.ckpt'

checkpoint_path13 = '/home/nextgen/Desktop/tf2/covid/training_1/cp7.ckpt'
checkpoint_path14 = '/home/nextgen/Desktop/tf2/covid/training_1/cp7.ckpt'

checkpoint_path15 = '/home/nextgen/Desktop/tf2/covid/training_1/cp8.ckpt'
checkpoint_path16 = '/home/nextgen/Desktop/tf2/covid/training_1/cp8.ckpt'


'''
my_model3.load_weights(checkpoint_path1)
my_model4.load_weights(checkpoint_path2)

result = my_model3.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)

result = my_model4.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)



my_model3.load_weights(checkpoint_path3)
my_model4.load_weights(checkpoint_path4)

result = my_model3.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)

result = my_model4.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)


my_model3.load_weights(checkpoint_path7)
my_model4.load_weights(checkpoint_path8)

result = my_model3.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)

result = my_model4.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)

'''
my_model4 = my_model3

my_model3.load_weights(checkpoint_path9)
my_model4.load_weights(checkpoint_path10)

result = my_model3.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)

result = my_model4.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)



my_model3.load_weights(checkpoint_path11)
my_model4.load_weights(checkpoint_path12)

result = my_model3.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)
result = my_model3.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)
print(result)

result = my_model4.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)


my_model3.load_weights('ssim_classification4.h5')

result = my_model3.evaluate(xr_test_rgb,labels_test_xr, batch_size=32)

print(result)


from sklearn.metrics import accuracy_score
pred = np.argmax(my_model3.predict(xr_test_rgb),axis=1)
test_label = np.argmax(labels_test_xr,axis=1)
acc = accuracy_score(test_label,pred)
print(acc)


my_model3.load_weights(checkpoint_path13)
my_model4.load_weights(checkpoint_path14)

result = my_model3.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)

result = my_model4.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)


my_model3.load_weights(checkpoint_path15)
my_model4.load_weights(checkpoint_path16)

result = my_model3.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)

result = my_model4.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)

my_model3.load_weights('ssim_classification5.h5')

result = my_model3.evaluate(xr_test_rgb,labels_test_xr, batch_size=32)

print(result)

my_model3.load_weights('ssim_classification6.h5')

result = my_model3.evaluate(xr_test_rgb,labels_test_xr, batch_size=32)

print(result)

my_model3.load_weights('ssim_classification7.h5')

result = my_model3.evaluate(xr_test_rgb,labels_test_xr, batch_size=32)

print(result)

####################################
##########metric
####################################
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

test_pred = my_model3.predict(xr_test_rgb)
test_pred = np.argmax(test_pred, axis=1)
test_labels = np.argmax(labels_test_xr, axis=1)

result = my_model3.evaluate(xr_test_rgb,labels_test_xr, batch_size = 32)
print(result)

confusion_matrix(test_labels, test_pred)
'''
array([[124,   1,   0],
       [  1, 121,   5],
       [  0,   1, 110]])
'''
total = 124+1+1+121+5+1+110
accuracy = (109+247)/total
tp = 124
fp = 1
fn = 1
tn = 237
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1score = 2*(precision*recall)/(precision+recall)
sensitivity = tp/(tp+fp)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

fpr, tpr, thresholds = roc_curve(test_labels,test_pred, pos_label=2)

auc(fpr, tpr)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(test_labels))[:, i], np.array(pd.get_dummies(test_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_auc

plt.imsave('/home/nextgen/Desktop/tf2/covid/orig_img_35.png', xr_test_rgb[35])
test_labels[129]
####################################
##########restored
####################################
unet_cbam_model = unet_cbam(pretrained_weights = 'nih_unet_cbam_20.h5', input_size=(512,512,3),kernel_size=3, ratio=3, activ_regularization=0.01 )

unet_cbam_model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3),
            loss='mse',
            metrics=['mse',ssim_loss])

print(unet_cbam_model.evaluate(xr_test_rgb,xr_test_rgb,batch_size=2))

restored = unet_cbam_model.predict(xr_test_rgb[150].reshape(-1,512,512,3))
plt.imshow((restored).reshape(512,512,3))

####################################
##########Grad cam, Score cam
####################################
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import gc

#tf.compat.v1.disable_eager_execution()

def normalize(x):
        """Utility function to normalize a tensor by its L2 norm"""
        return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

def GradCam(model, img_array, layer_name):
    cls = np.argmax(model.predict(img_array))
    
    """GradCAM method for visualizing input saliency."""
    y_c = model.output[0, cls]
    conv_output = model.get_layer(layer_name).output
    grads = tf.gradients(y_c, conv_output)[0]
    # grads = normalize(grads)

    gradient_function = K.function([model.input], [conv_output, grads])
    output, grads_val = gradient_function([img_array])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    weights = np.mean(grads_val, axis=(0, 1))

    cam = np.dot(output, weights)
    cam = np.maximum(cam, 0)  # Passing through ReLU
    cam /= np.max(cam)  # scale 0 to 1.0  

    return cam

def GradCamPlusPlus(model, img_array, layer_name):
    cls = np.argmax(model.predict(img_array))
    y_c = model.output[0, cls]
    conv_output = model.get_layer(layer_name).output
    grads = tf.gradients(y_c, conv_output)[0]
    # grads = normalize(grads)

    first = K.exp(y_c)*grads
    second = K.exp(y_c)*grads*grads
    third = K.exp(y_c)*grads*grads*grads

    gradient_function = K.function([model.input], [y_c,first,second,third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([img_array])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)
    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)
    alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))
    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)

    cam = np.sum(deep_linearization_weights*conv_output[0], axis=2)
    cam = np.maximum(cam, 0)  # Passing through ReLU
    cam /= np.max(cam) # scale 0 to 1.0  

    return cam

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

def ScoreCam(model, img_array, layer_name, max_N=-1):

    cls = np.argmax(model.predict(img_array))
    act_map_array = Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(img_array)
    
    # extract effective maps
    if max_N != -1:
        act_map_std_list = [np.std(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])]
        unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
        max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
        act_map_array = act_map_array[:,:,:,max_N_indices]

    input_shape = model.layers[0].input_shape[1:]  # get input shape
    # 1. upsampled to original input size
    act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape[:2], interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
    # 2. normalize the raw activation value in each activation map into [0, 1]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)
    # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(img_array)
        for k in range(3):
            masked_input[0,:,:,k] *= act_map_normalized
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)
    # 4. feed masked inputs into CNN model and softmax
    pred_from_masked_input_array = softmax(model.predict(masked_input_array))
    # 5. define weight as the score of target class
    weights = pred_from_masked_input_array[:,cls]
    # 6. get final class discriminative localization map as linear weighted combination of all activation maps
    cam = np.dot( act_map_array[0,:,:,:], weights)
    cam = np.maximum(0, cam)  # Passing through ReLU
    cam /= np.max(cam)  # scale 0 to 1.0
    
    return cam

def superimpose(original_img_path, cam, emphasize=False):
    
    img_bgr = original_img_path

    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, 0.5, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    hif = 0.5
    superimposed_img = heatmap * hif + img_bgr*255
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img_rgb

import tensorflow.keras

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras.applications.vgg16 import preprocess_input

def build_guided_model(build_model_function):
    """Function returning modified model.
    
    Changes gradient function for all ReLu activations according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model_function()
    return new_model

def GuidedBackPropagation(model, img_array, layer_name):
    model_input = model.input
    layer_output = model.get_layer(layer_name).output
    max_output = K.max(layer_output, axis=3)
    grads = tf.gradients(max_output, model_input)[0]
    get_output = K.function([model_input], [grads])
    saliency = get_output([img_array])
    saliency = np.clip(saliency[0][0], 0.0, 1.0)  # scale 0 to 1.0  
    return saliency

def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

from tensorflow.keras.preprocessing.image import load_img, img_to_array

def read_and_preprocess_img(path, size=(224,224)):
    img = load_img(path, target_size=size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

for layer in my_model3.layers:

    print(layer.name)
#conv2d_33

layer_name = 'conv2d_33'
#80 90 95 195 20
orig_img=xr_test_rgb[23]
img_bgr=orig_img
img_array = xr_test_rgb[23].reshape(-1,512,512,3)
#rad_cam = GradCam(my_model3,xr_test_rgb[0].reshape(-1,512,512,3),layer_name)
score_cam = ScoreCam(my_model3,orig_img.reshape(-1,512,512,3),layer_name,max_N=-1)
score_cam_resized = cv2.resize(score_cam, (orig_img.shape[1], orig_img.shape[0]))
plt.imshow(score_cam_resized)

score_cam_superimposed = superimpose(orig_img, score_cam)
plt.imshow(score_cam_superimposed)
plt.imshow(orig_img)

orig_img = np.float32(orig_img)
img_gray = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
#img_grya=orig_img
dx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
dy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
grad = np.sqrt(dx ** 2 + dy ** 2) 
grad = cv2.dilate(grad,kernel=np.ones((5,5)), iterations=1)  
grad -= np.min(grad)
grad /= np.max(grad)
grad_times_score_cam = grad * score_cam_resized
plt.imshow(grad_times_score_cam)
#labels_test_xr[272] pneumonial 36 60 79 83! 111!
#labels_test_xr[57] covid 58 115! 194!209!
labels_test_xr[206]
plt.imsave('/home/nextgen/Desktop/tf2/covid/orig_%d.png'%(83), xr_test_rgb[83])


heatmap = (score_cam_superimposed - score_cam_superimposed.min()) / (score_cam_superimposed.max() - score_cam_superimposed.min())
cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
output_image = cv2.addWeighted(cv2.cvtColor(orig_img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
output_image = cv2.addWeighted(cv2.cvtColor(orig_img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.1, score_cam_superimposed, 1, 0)

output_image = np.hstack((orig_img, score_cam))
plt.imshow(orig_img+score_cam_superimposed)
plt.imshow(output_image)


##github
score_cam=ScoreCam(model,img_array,layer_name)
score_cam_superimposed = superimpose(img_path, score_cam)
score_cam_emphasized = superimpose(img_path, score_cam, emphasize=True)








for i in range(len(xr_test_rgb)):
    orig_img = xr_test_rgb[i]
    score_cam = ScoreCam(my_model3,orig_img.reshape(-1,512,512,3),layer_name,max_N=-1)
    score_cam_resized = cv2.resize(score_cam, (orig_img.shape[1], orig_img.shape[0]))
    score_cam_superimposed = superimpose(orig_img ,score_cam)
    plt.imsave('/home/nextgen/Desktop/tf2/covid/scorecam2/score_cam_%d.png'%(i), score_cam_superimposed)

for i in range(len(xr_test_rgb)):
    orig_img = xr_test_rgb[i]
    score_cam = ScoreCam(my_model3,orig_img.reshape(-1,512,512,3),layer_name,max_N=-1)
    score_cam_resized = cv2.resize(score_cam, (orig_img.shape[1], orig_img.shape[0]))
    orig_img = np.float32(orig_img)
    img_gray = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
    #img_grya=orig_img
    dx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(dx ** 2 + dy ** 2) 
    grad = cv2.dilate(grad,kernel=np.ones((5,5)), iterations=1)  
    grad -= np.min(grad)
    grad /= np.max(grad)  
    grad_times_score_cam = grad * score_cam_resized
    plt.imsave('/home/nextgen/Desktop/tf2/covid/grad_tiems_score_cam/score_cam_%d.png'%(i), grad_times_score_cam)



fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(11, 11))
ax[0,0].imshow(orig_img)
ax[0,1].imshow(grad_cam_superimposed)


plt.figure()
plt.imshow(score_cam_resized+orig_img)
plt.imshow(orig_img)
plt.close()

plt.imsave('/home/nextgen/Desktop/tf2/covid/test_134.png',xr_test_rgb[134])




#################################################
####laod xlsx
#################################################
sc = pd.read_excel('/home/nextgen/Desktop/tf2/covid/logits_test.xlsx')
test_preds = pd.read_excel('/home/nextgen/Desktop/tf2/covid/pred_classes_test.xlsx')
labels_test = pd.read_excel('/home/nextgen/Desktop/tf2/covid/labaels_test.xlsx')

sc = sc.iloc[:,1:]
test_preds = test_preds.iloc[:,1:]
test_labels = labels_test.iloc[:,1:]

############################################
####confusion matrix
############################################
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

sc = my_model3.predict(xr_test_rgb)

test_preds = my_model3.predict_classes(xr_test_rgb)

test_labels = np.argmax(labels_test_xr, axis=1)


confusion_matrix(test_labels,test_preds)

accuracy_score(test_labels,test_pred)

fpr, tpr, thresholds = roc_curve(test_labels,test_pred, pos_label=2)

auc(fpr, tpr)

f1_score(test_labels,test_pred, average='micro')

'''
array([[125,   2,   0],
       [  0, 121,   2],
       [  0,   4, 109]])
'''

'''
array([[125,   0,   0],
       [  1, 125,   1],
       [  1,   2, 108]])
'''

#pneumonia
total = 125+1+125+1+2+2+107
accuracy = (125+125+108)/total
tp = 125
fp = 0
fn= 2
tn = 236
precision = tp/(tp+fp)
#normal 0.9908
recall = tp/(tp+fn)
#normal 1.0
f1score = 2*(precision*recall)/(precision+recall)
f1score
#normal 0.9954
sensitivity = tp/(tp+fp)
#normal 0.9908
specificity = tn/(tn+fp)
# normal 0.9960
accuracy = (tp+tn)/(tp+tn+fp+fn)
#normal 0.9972
tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

################confusion matrix
import seaborn as sn
conf = np.array([[125/125,   0,   0],
       [  1/127, 125/127,   1/127],
       [  1/111,   2/111, 108/111]])


df_conf = pd.DataFrame(conf, index = ['COVID19', 'Pneumonia', 'Normal'], columns = ['COVID19', 'Pneumonia', 'Normal'])
cmap = plt.get_cmap('Blues')
plt.imshow(df_conf, interpolation='nearest', cmap=cmap)
plt.matshow(df_conf, cmap=cmap)
plt.show()

sn.set(font_scale=1)
sn.heatmap(df_conf, annot=True, annot_kws={'size':12})
plt.show()



################roc curve

# Compute ROC curve and ROC area for each class
from itertools import cycle
lw =2
n_classes=3
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(labels_test_xr[:, i], sc[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(labels_test_xr.ravel(), sc.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
n_classes = 3
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
label= ['COVID-19', 'Pneumonia', 'Normal']
# Plot all ROC curves
plt.figure()
lb = cycle(['COVID-19','Pneumonia','Normal'])
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label= '{0} = {1:0.4f}'
             ''.format(label[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()