#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:26:00 2021

@author: nextgen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:20:21 2021

@author: nextgen
"""

import os

import configs

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)


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
mypath = '/home/ubuntu/Desktop/data/pjh/images/images'
targetdir= '/home/ubuntu/Desktop/data/pjh/images/images'
fileExt = r'.png'
onlyfiles = [os.path.join(mypath, _) for _ in os.listdir(mypath) if _.endswith(fileExt)]
onlyfiles1 = onlyfiles[0:-1000]
onlyfiles2 = onlyfiles[-1001:-1]


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

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    conv1 = CBAM_attention(conv1, ratio, kernel_size, dr_ratio, activ_regularization)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)

    conv2 = CBAM_attention(conv2, ratio, kernel_size, dr_ratio, activ_regularization)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

    conv3 = CBAM_attention(conv3, ratio, kernel_size, dr_ratio, activ_regularization)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    conv4 = CBAM_attention(conv4, ratio, kernel_size, dr_ratio, activ_regularization)

    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(drop5))

    merge6 = concatenate([drop4, up6], axis=3)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    conv6 = CBAM_attention(conv6, ratio, kernel_size, dr_ratio, activ_regularization)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv6))

    merge7 = concatenate([conv3, up7], axis=3)

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    conv7 = CBAM_attention(conv7, ratio, kernel_size, dr_ratio, activ_regularization)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv7))

    merge8 = concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv8))

    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)

    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='relu')(conv9)

    final = UpSampling3D(size=(1, 1, 3))(conv10)

    model = Model(inputs=inputs, outputs=final)

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

dr_ratio = 0.2
ratio=8
activ_regularization=0.0001
kernel_size=7
kernel_initializer = tf.keras.initializers.VarianceScaling()
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def CBAM_attention(inputs,ratio,kernel_size,dr_ratio,activ_regularization):
    x = inputs
    channel = x.get_shape()[-1]

    ##channel attention##
    avg_pool = tf.reduce_mean(x, axis=[1,2], keepdims=True)
    avg_pool = Dense(units = channel//ratio ,activation='relu', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), use_bias=True,bias_initializer='zeros',trainable=True)(avg_pool)
    avg_pool = Dense(channel, activation = 'relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True, bias_initializer='zeros',trainable=True)(avg_pool)

    max_pool = tf.reduce_max(x, axis=[1,2], keepdims=True)
    max_pool = Dense(units = channel//ratio, activation='relu', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), use_bias=True,bias_initializer='zeros',trainable=True)(max_pool)
    max_pool = Dense(channel, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), activity_regularizer=tf.keras.regularizers.l1(activ_regularization),use_bias=True, bias_initializer='zeros',trainable=True)(max_pool)
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


checkpoint_path1 = './pretrained_weights_nih'
checkpoint_dir1 = os.path.join(os.getcwd()+checkpoint_path1)
checkpoint_path2 = './pretrained_weights'
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
'''
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    unet_cbam_model = unet(pretrained_weights = None, input_size=(512,512,3),kernel_size=3, ratio=3, activ_regularization=0.01 )
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
'''


print('------------------loading data--------------------')

#data
xr_train_rgb = np.load('xr_train.npy')
xr_valid_rgb = np.load('xr_valid.npy')
xr_test_rgb = np.load('xr_test.npy')
labels_train_xr = np.load('labels_train_xr.npy')
labels_valid_xr = np.load('labels_valid_xr.npy')
labels_test_xr = np.load('labels_test_xr.npy')

print('------------------building model--------------------')

'''
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    unet_cbam_model = unet_cbam(pretrained_weights = 'nih_unet_cbam_12.h5', input_size=(512,512,3),kernel_size=3, ratio=3, activ_regularization=0.01 )

    unet_cbam_classification = Model(inputs = unet_cbam_model.input, outputs= unet_cbam_model.get_layer('dropout_1').output)

    my_model3 = tf.keras.Sequential()

    my_model3.add(unet_cbam_classification)

    my_model3.add(Conv2D(256,2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))

    #my_model2.add(cbam_tf.cbam_block(my_model2.output,name=None, ratio=2))

    my_model3.add(Conv2D(128,2, activation = 'relu', padding = 'same',  kernel_initializer = 'he_normal'))

    #my_model2.add(cbam_tf.cbam_block(my_model2.output,name=None,ratio=2))

    my_model3.add(MaxPooling2D(pool_size=(2,2)))

    my_model3.add(GlobalAveragePooling2D())

    my_model3.add(Dropout(0.3))

    my_model3.add(Dense(128,  activation = 'relu'))

    my_model3.add(Dropout(0.5))

    my_model3.add(Dense(64,  activation='relu'))

    #my_model3.add(Dropout(0.5))

    my_model3.add(Dense(32, activation='relu' ))

    #my_model3.add(Dropout(0.5))

    my_model3.add(Dense(16, activation='relu' ))

    my_model3.add(Dense(8, activation = 'relu' ))

    #my_model3.add(Dropout(0.3))

    my_model3.add(BatchNormalization())

    my_model3.add(Dense(3, activation = 'softmax'))

    my_model3.layers[0].trainable = True

    my_model3.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])


'''
activ_regularization=0.00001

unet_cbam_model = unet_cbam(pretrained_weights = 'unet_xr_cbam.h5', input_size=(512,512,3))

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

#my_model3.add(Dropout(0.3))

my_model3.add(Dense(128, activation = 'relu', activity_regularizer=tf.keras.regularizers.l2(activ_regularization),kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

my_model3.add(Dropout(0.5))

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

my_model3.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.0001, clipvalue=2, clipnorm=1),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

print(my_model3.summary())
checkpoint_path1 = './training_1/cp7.ckpt'
checkpoint_path1 = './pretrained_weights_nih.ckpt'
#my_model3.load_weights(checkpoint_path1 )
#my_model3.load_weights('ssim_classification6.h5')

#my_model3.load_weights('ssim_classification1.h5')

def cross_entropy(y_true, y_pred):
    return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(tf.clip_by_value(y_pred,1e-10,1.0))))
    #return tf.reduce_mean(tf.image.ssim(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), 2.0))

def compute_loss(y_true, y_pred):
  per_example_loss = tf.keras.losses.CategoricalCrossentropy(y_true, y_pred)
  per_example_loss /= tf.cast(
      tf.shape(y_true)[1] * tf.shape(y_true)[2], tf.float32)  # <--- See if this fixes the NaNs.
  return tf.nn.compute_average_loss(
      per_example_loss, global_batch_size=16)





from tensorflow.keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(rotation_range = 20,
                        zoom_range = 0.15,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.15,
                        horizontal_flip=True,
                        fill_mode='nearest')

print('------------------create callbacks--------------------')

#checkpoint_path1 = './training_1/cp_unet2.ckpt'
checkpoint_dir2 = os.path.join(os.getcwd()+checkpoint_path1)
checkpoint_path2 = './pretrained_weights/cp_unet2.ckpt'
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
    (15, 0.0001),
    (50, 0.0001),
    (60, 0.0001),
    (70, 0.0001),
    (80, 0.00008),
    (130, 0.00005),
    (150, 0.0001),
    (200, 0.00005),
    (250, 0.00003),
    (280, 0.00002),
    (300, 0.00005),
    (350, 0.00003),
    (400, 0.00002),
    (450, 0.00002),
    (700, 0.00001)
]


from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=8, verbose=1, mode='min', min_lr=5e-7)

my_model3.load_weights(checkpoint_path1)
print('------------------fitting----------------------------')
my_history = my_model3.fit(aug.flow(xr_train_rgb, np.array(labels_train_xr), batch_size=16),
	validation_data=(xr_valid_rgb,np.array(labels_valid_xr)),
    steps_per_epoch=xr_train_rgb.shape[0]//16, # number of images comprising of one epoch
    validation_steps=xr_valid_rgb.shape[0]//16,
    callbacks=[cp_callback1,cp_callback2,CustomLearningRateScheduler(lr_schedule)],
	epochs= 200)

result = my_model3.evaluate(xr_test_rgb,labels_test_xr,batch_size=32)

print(result)

print('------------------saving weights------------------------')
my_model3.save_weights('ssim_classification7.h5')
print('------------------weights saved------------------------')

plt.figure()
plt.plot(my_history.history['loss'])
plt.plot(my_history.history['val_loss'])
plt.savefig('train_result.png')
plt.close()




'''
ssim_classification4.h5 -> 0.9835 in test set
'''



'''

from tensorflow.keras import optimizers
learning_rate = 0.0001

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
    callbacks=[cp_callback1,cp_callback2],#CustomLearningRateScheduler(lr_schedule)],
	epochs=50)
'''