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

def ssim_loss_minusone(y_true, y_pred):
    return 1-tf.reduce_mean(tf.image.ssim(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), 2.0))

def ssim_loss(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), 2.0))
