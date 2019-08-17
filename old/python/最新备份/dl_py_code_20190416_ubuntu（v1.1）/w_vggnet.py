from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image    #image：用于图像数据的实时数据增强的（相当）基本的工具集。
import numpy as np
import matplotlib.pyplot as plt
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay,visualize_cam,visualize_activation
from keras import activations
import matplotlib.cm as cm
import cv2
import h5py
import os
import random
import matplotlib.image as mpimg
import scipy.misc as misc

from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Conv3D, MaxPooling3D
from keras.optimizers import SGD














def vgg16_w_3d(classes):
    inputs = Input(shape=(280, 280, 16, 1), name='input')
    # Block 1
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block5_pool')(x)

    # dense
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(2021, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x, name='vgg16')

    return model

