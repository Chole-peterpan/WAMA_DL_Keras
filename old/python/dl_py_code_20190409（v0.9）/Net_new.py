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
    GlobalAveragePooling3D,Dense,Flatten,Lambda,Dropout,Activation,ZeroPadding3D,multiply,LeakyReLU
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as keras
from keras.layers.core import Reshape
from keras.backend import int_shape
from keras.layers.merge import concatenate
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

def squeeze_excite_block3d(input, ratio=2):
    nb_channel = int_shape(input)[-1]
    se_shape = (1, 1, 1, nb_channel)

    out = GlobalAveragePooling3D()(input)
    out = Reshape(se_shape)(out)
    out = Dense(nb_channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(out)  # //表示相除并取整
    out = Dense(nb_channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(out)

    out = multiply([input, out])

    return out

def identity_block(x, nb_filters, name, kernel_size=3):
    k1, k2, k3 = nb_filters
    convname1 = name + 'conv1'
    convname2 = name + 'conv2'
    convname3 = name + 'conv3'
    bnname1 = name + 'bn1'
    bnname2 = name + 'bn2'
    bnname3 = name + 'bn3'
    out = Conv3D(k1, 1, strides=1, kernel_initializer='he_normal', name = convname1)(x)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname1)(out)
    out = Activation('relu')(out)

    out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname2)(out)
    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name = convname3)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname3)(out)

    out = merge([out, x], mode='sum')
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def se_identity_block(x, nb_filters, name, kernel_size=3):
    k1, k2, k3 = nb_filters
    convname1 = name + 'conv1'
    convname2 = name + 'conv2'
    convname3 = name + 'conv3'
    bnname1 = name + 'bn1'
    bnname2 = name + 'bn2'
    bnname3 = name + 'bn3'
    out = Conv3D(k1, 1, strides=1, kernel_initializer='he_normal', name = convname1)(x)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname1)(out)
    out = Activation('relu')(out)

    out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname2)(out)
    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name = convname3)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname3)(out)

    out = squeeze_excite_block3d(out)

    out = merge([out, x], mode='sum')
    out = BatchNormalization()(out)
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
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname1)(out)
    out = Activation('relu')(out)

    out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6, name = bnname2)(out)
    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name = convname3)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname3)(out)

    x = Conv3D(k3, 2, strides=2, kernel_initializer='he_normal', name = convname4)(x)
    x = BatchNormalization(axis = -1, epsilon = 1e-6, name = bnname4)(x)

    out = merge([out, x], mode='sum')
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def ClassNet():
    inputs = Input(shape=(280, 280, 16, 1), name='input1')
    # 256*256*128
    print("input shape:", inputs.shape)  # (?, 140, 140, 16, 64)
    out = Conv3D(64, 7, strides=(2, 2, 1), padding = 'same', kernel_initializer='he_normal', use_bias = False, name = 'conv1')(inputs)
    print("conv0 shape:", out.shape)#(?, 140, 140, 16, 64)
    out = BatchNormalization(axis = -1, epsilon = 1e-6, name = 'bn1')(out)
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

def se_ClassNet():
    inputs = Input(shape=(280, 280, 16, 1), name='input1')
    # 256*256*128
    print("input shape:", inputs.shape)  # (?, 140, 140, 16, 64)
    out = Conv3D(64, 7, strides=(2, 2, 1), padding = 'same', kernel_initializer='he_normal', use_bias = False, name = 'conv1')(inputs)
    print("conv0 shape:", out.shape)#(?, 140, 140, 16, 64)
    out = BatchNormalization(axis = -1, epsilon = 1e-6, name = 'bn1')(out)
    out = Activation('relu')(out)
    out = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding = 'same')(out)
    print("pooling1 shape:", out.shape)#(?, 70, 70, 16, 64)


    # stage1=================================================
    out = conv_block(out, [64, 64, 256], name = 'L1_block1')
    print("conv1 shape:", out.shape)
    out = se_identity_block(out, [64, 64, 256], name = 'L1_block2')
    out = se_identity_block(out, [64, 64, 256], name = 'L1_block3')

    # stage2=================================================
    out = conv_block(out, [128, 128, 512], name = 'L2_block1')
    print("conv2 shape:", out.shape)
    out = se_identity_block(out, [128, 128, 512], name = 'L2_block2')
    out = se_identity_block(out, [128, 128, 512], name = 'L2_block3')
    out = se_identity_block(out, [128, 128, 512], name = 'L2_block4')

    # stage3=================================================
    out = conv_block(out, [256, 256, 1024], name = 'L3_block1')
    print("conv3 shape:", out.shape)
    out = se_identity_block(out, [256, 256, 1024], name = 'L3_block2')
    out = se_identity_block(out, [256, 256, 1024], name = 'L3_block3')
    out = se_identity_block(out, [256, 256, 1024], name = 'L3_block4')
    out = se_identity_block(out, [256, 256, 1024], name = 'L3_block5')
    out = se_identity_block(out, [256, 256, 1024], name = 'L3_block6')

    # stage4=================================================
    out = conv_block(out, [512, 512, 2048], name = 'L4_block1')
    print("conv4 shape:", out.shape)
    out = se_identity_block(out, [512, 512, 2048], name = 'L4_block2')
    out = se_identity_block(out, [512, 512, 2048], name = 'L4_block3')

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



# multi_input_net
# 可有多输入的resnet
# input1:动脉期 a
# input2:门脉期 v
def multi_input_ClassNet():


    # 4 input1:a =======================================================================================================
    inputs_1 = Input(shape=(280, 280, 16, 1), name='path1_input1')
    # 256*256*128
    print("path1_input shape:", inputs_1.shape)  # (?, 140, 140, 16, 64)
    out1 = Conv3D(64, 7, strides=(2, 2, 1), padding = 'same', kernel_initializer='he_normal', use_bias = False, name = 'path1_conv1')(inputs_1)
    print("path1_conv0 shape:", out1.shape)#(?, 140, 140, 16, 64)
    out1 = BatchNormalization(axis = -1, epsilon = 1e-6, name = 'path1_bn1')(out1)
    out1 = Activation('relu')(out1)
    out1 = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding = 'same')(out1)
    print("path1_pooling1 shape:", out1.shape)#(?, 70, 70, 16, 64)

    out1 = conv_block(out1, [64, 64, 256], name = 'path1_L1_block1')
    print("path1_conv1 shape:", out1.shape)
    out1 = identity_block(out1, [64, 64, 256], name = 'path1_L1_block2')
    out1 = identity_block(out1, [64, 64, 256], name = 'path1_L1_block3')

    # 4 input2:v =======================================================================================================
    inputs_2 = Input(shape=(280, 280, 16, 1), name='path2_input2')
    # 256*256*128
    print("path2_input shape:", inputs_2.shape)  # (?, 140, 140, 16, 64)
    out2 = Conv3D(64, 7, strides=(2, 2, 1), padding = 'same', kernel_initializer='he_normal', use_bias = False, name = 'path2_conv1')(inputs_2)
    print("path2_conv0 shape:", out1.shape)#(?, 140, 140, 16, 64)
    out2 = BatchNormalization(axis = -1, epsilon = 1e-6, name = 'path2_bn1')(out2)
    out2 = Activation('relu')(out2)
    out2 = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding = 'same')(out2)
    print("path2_pooling1 shape:", out1.shape)#(?, 70, 70, 16, 64)

    out2 = conv_block(out2, [64, 64, 256], name = 'path2_L1_block1')
    print("path2_conv1 shape:", out2.shape)
    out2 = identity_block(out2, [64, 64, 256], name = 'path2_L1_block2')
    out2 = identity_block(out2, [64, 64, 256], name = 'path2_L1_block3')


    #main path:concatenate 'out1' and 'out2' into 'out' ================================================================
    out = concatenate([out1, out2], axis=-1)
    print("concatenate shape:", out.shape)



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
    output = Activation(activation = 'sigmoid')(out)

    model = Model(input = [inputs_1, inputs_2], output = output)
    #mean_squared_logarithmic_error or binary_crossentropy
    model.compile(optimizer=SGD(lr = 1e-6, momentum = 0.9), loss = EuiLoss, metrics= [y_t, y_pre, Acc] )

    print('im multi_input_ClassNet')
    return model

# leaky relu 版本 =========================================================================
def identity_block_lk(x, nb_filters, name, kernel_size=3):
    k1, k2, k3 = nb_filters
    convname1 = name + 'conv1'
    convname2 = name + 'conv2'
    convname3 = name + 'conv3'
    bnname1 = name + 'bn1'
    bnname2 = name + 'bn2'
    bnname3 = name + 'bn3'
    out = Conv3D(k1, 1, strides=1, kernel_initializer='he_normal', name = convname1)(x)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname1)(out)
    out = LeakyReLU(alpha = 0.2)(out)

    out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname2)(out)
    out = LeakyReLU(alpha=0.2)(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name = convname3)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname3)(out)

    out = merge([out, x], mode='sum')
    out = BatchNormalization()(out)
    out = LeakyReLU(alpha=0.2)(out)
    return out

def conv_block_lk(x, nb_filters, name, kernel_size=3):
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
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname1)(out)
    out = LeakyReLU(alpha=0.2)(out)

    out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6, name = bnname2)(out)
    out = LeakyReLU(alpha=0.2)(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name = convname3)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname3)(out)

    x = Conv3D(k3, 2, strides=2, kernel_initializer='he_normal', name = convname4)(x)
    x = BatchNormalization(axis = -1, epsilon = 1e-6, name = bnname4)(x)

    out = merge([out, x], mode='sum')
    out = BatchNormalization()(out)
    out = LeakyReLU(alpha=0.2)(out)
    return out


def multi_input_ClassNet_lk():


    # 4 input1:a =======================================================================================================
    inputs_1 = Input(shape=(280, 280, 16, 1), name='path1_input1')
    # 256*256*128
    print("path1_input shape:", inputs_1.shape)  # (?, 140, 140, 16, 64)
    out1 = Conv3D(64, 7, strides=(2, 2, 1), padding = 'same', kernel_initializer='he_normal', use_bias = False, name = 'path1_conv1')(inputs_1)
    print("path1_conv0 shape:", out1.shape)#(?, 140, 140, 16, 64)
    out1 = BatchNormalization(axis = -1, epsilon = 1e-6, name = 'path1_bn1')(out1)
    out1 = LeakyReLU(alpha=0.2)(out1)#=====================
    out1 = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding = 'same')(out1)
    print("path1_pooling1 shape:", out1.shape)#(?, 70, 70, 16, 64)

    out1 = conv_block(out1, [64, 64, 256], name = 'path1_L1_block1')
    print("path1_conv1 shape:", out1.shape)
    out1 = identity_block(out1, [64, 64, 256], name = 'path1_L1_block2')
    out1 = identity_block(out1, [64, 64, 256], name = 'path1_L1_block3')

    # 4 input2:v =======================================================================================================
    inputs_2 = Input(shape=(280, 280, 16, 1), name='path2_input2')
    # 256*256*128
    print("path2_input shape:", inputs_2.shape)  # (?, 140, 140, 16, 64)
    out2 = Conv3D(64, 7, strides=(2, 2, 1), padding = 'same', kernel_initializer='he_normal', use_bias = False, name = 'path2_conv1')(inputs_2)
    print("path2_conv0 shape:", out1.shape)#(?, 140, 140, 16, 64)
    out2 = BatchNormalization(axis = -1, epsilon = 1e-6, name = 'path2_bn1')(out2)
    out2 = LeakyReLU(alpha=0.2)(out2)  # =====================
    out2 = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding = 'same')(out2)
    print("path2_pooling1 shape:", out1.shape)#(?, 70, 70, 16, 64)

    out2 = conv_block(out2, [64, 64, 256], name = 'path2_L1_block1')
    print("path2_conv1 shape:", out2.shape)
    out2 = identity_block(out2, [64, 64, 256], name = 'path2_L1_block2')
    out2 = identity_block(out2, [64, 64, 256], name = 'path2_L1_block3')


    #main path:concatenate 'out1' and 'out2' into 'out' ================================================================
    out = concatenate([out1, out2], axis=-1)
    print("concatenate shape:", out.shape)



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
    output = Activation(activation = 'sigmoid')(out)

    model = Model(input = [inputs_1, inputs_2], output = output)
    #mean_squared_logarithmic_error or binary_crossentropy
    model.compile(optimizer=SGD(lr = 1e-6, momentum = 0.9), loss = EuiLoss, metrics= [y_t, y_pre, Acc] )

    print('im multi_input_ClassNet_lk')
    return model


