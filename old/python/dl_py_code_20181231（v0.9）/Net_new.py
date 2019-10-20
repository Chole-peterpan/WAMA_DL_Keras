# import cv2
import numpy as np
import random
import h5py
from matplotlib import pyplot as plt
import os
from PIL import Image
import math
import PIL

from keras.models import *
from keras.layers import Input, merge,Conv3D,BatchNormalization,Conv3DTranspose, UpSampling3D, MaxPooling3D, AveragePooling3D, \
    GlobalAveragePooling3D,Dense,Flatten,Lambda,Dropout,Activation,ZeroPadding3D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as keras
from keras.utils.vis_utils import plot_model
from keras import backend as k
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



def Acc(y_true, y_pred):
    y_pred_r = K.round(y_pred)
    return K.equal(y_pred_r, y_true)

def y_t(y_true, y_pred):
    return y_true

def y_pre(y_true, y_pred):
    return y_pred

def EuiLoss(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    d = K.sum(K.sqrt(K.square(y_true_f - y_pred_f) + 1e-12))
    a = K.cast(K.greater_equal(d, 0.5), dtype='float32')
    b = K.cast(K.greater_equal(0.12, d), dtype='float32')
    c = K.cast(K.greater_equal(0.3, d), dtype='float32')
    loss = (2 + 4 * a - 0.5 * b - 1 * c) * d + 0.2 * y_pred_f *d
    return loss

def identity_block(x, nb_filters, name, kernel_size=3):
    k1, k2, k3 = nb_filters
    convname1 = name + 'conv1'
    convname2 = name + 'conv2'
    convname3 = name + 'conv3'
    bnname1 = name + 'bn1'
    bnname2 = name + 'bn2'
    bnname3 = name + 'bn3'
    out = Conv3D(k1, 1, strides=1, kernel_initializer='he_normal', name = convname1)(x)
#    out = BatchNormalization(axis = -1, epsilon = 1e-6, trainable = is_train, name = bnname1)(out)
    out = Activation('relu')(out)

    out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2)(out)
#    out = BatchNormalization(axis = -1, epsilon = 1e-6, trainable = is_train, name = bnname2)(out)
    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name = convname3)(out)
#    out = BatchNormalization(axis = -1, epsilon = 1e-6, trainable = is_train, name = bnname3)(out)

    out = merge([out, x], mode='sum')
#    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def conv_block(x, nb_filters, name, kernel_size=3):
    k1, k2, k3 = nb_filters
    convname1 = name + 'conv1'
    convname2 = name + 'conv2'
    convname3 = name + 'conv3'
    convname4 = name + 'conv4'
    bnname1 = name + 'bn1'
    bnname2 = name + 'bn2'
    bnname3 = name + 'bn3'
    bnname4 = name + 'bn4'
    out = Conv3D(k1, 2, strides=2, kernel_initializer='he_normal', name = convname1)(x)
#    out = BatchNormalization(axis = -1, epsilon = 1e-6, trainable = is_train, name = bnname1)(out)
    out = Activation('relu')(out)

    out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2)(out)
#    out = BatchNormalization(axis = -1, epsilon = 1e-6, trainable = is_train, name = bnname2)(out)
    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name = convname3)(out)
#    out = BatchNormalization(axis = -1, epsilon = 1e-6, trainable = is_train, name = bnname3)(out)

    x = Conv3D(k3, 2, strides=2, kernel_initializer='he_normal', name = convname4)(x)
#    x = BatchNormalization(axis = -1, epsilon = 1e-6, trainable = is_train, name = bnname4)(x)

    out = merge([out, x], mode='sum')
    #out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def ClassNet():
    inputs = Input(shape=(280, 280, 16, 1), name='input1')
    # 256*256*128
    print("input shape:", inputs.shape)  # (?, 140, 140, 16, 64)
    out = Conv3D(64, 7, strides=(2, 2, 1), padding = 'same', kernel_initializer='he_normal', use_bias = False, name = 'conv1')(inputs)
    print("conv0 shape:", out.shape)#(?, 140, 140, 16, 64)
#    out = BatchNormalization(axis = -1, epsilon = 1e-6, trainable = is_train, name = 'bn1')(out)
    out = Activation('relu')(out)
    out = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding = 'same')(out)
    print("pooling1 shape:", out.shape)#(?, 70, 70, 16, 64)

    out = conv_block(out, [64, 64, 256], name = 'L1_block1')
    print("conv1 shape:", out.shape)
    out = identity_block(out, [64, 64, 256], name = 'L1_block2')

    out = identity_block(out, [64, 64, 256], name = 'L1_block3')


    out = conv_block(out, [128, 128, 512], name = 'L2_block1')
    print("conv2 shape:", out.shape)
    out = identity_block(out, [128, 128, 512], name = 'L2_block2')

    out = identity_block(out, [128, 128, 512], name = 'L2_block3')

    out = identity_block(out, [128, 128, 512], name = 'L2_block4')


    out = conv_block(out, [256, 256, 1024], name = 'L3_block1')
    print("conv3 shape:", out.shape)
    out = identity_block(out, [256, 256, 1024], name = 'L3_block2')
    out = identity_block(out, [256, 256, 1024], name = 'L3_block3')
    out = identity_block(out, [256, 256, 1024], name = 'L3_block4')
    out = identity_block(out, [256, 256, 1024], name = 'L3_block5')
    out = identity_block(out, [256, 256, 1024], name = 'L3_block6')

    out = conv_block(out, [512, 512, 2048], name = 'L4_block1')
    print("conv4 shape:", out.shape)
    out = identity_block(out, [512, 512, 2048], name = 'L4_block2')
    out = identity_block(out, [512, 512, 2048], name = 'L4_block3')

    out = GlobalAveragePooling3D(data_format = 'channels_last')(out)
    print("Gpooling shape:", out.shape)
    out_drop = Dropout(rate=0.3)(out)
    out = Dense(1, name = 'fc1')(out_drop)
    print("out shape:", out.shape)
    #out = Dense(1, name = 'fc1')(out)
    output = Activation(activation = 'sigmoid')(out)

    model = Model(input = inputs, output = output)
    #mean_squared_logarithmic_error or binary_crossentropy
    model.compile(optimizer=SGD(lr = 1e-6, momentum = 0.9), loss = EuiLoss, metrics= [y_t, y_pre, Acc] )
    return model