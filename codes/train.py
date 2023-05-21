import os
import configs
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
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
