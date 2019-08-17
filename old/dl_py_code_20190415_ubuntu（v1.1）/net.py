from keras.layers import merge, GlobalAveragePooling3D, multiply, Conv3D,Add
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D, AveragePooling3D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input

import os
import numpy as np
import random, h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.backend import int_shape

def squeeze_excite_block3d(input, ratio = 2):
    nb_channel = int_shape(input)[-1]
    SE_shape = (1, 1, 1, nb_channel)

    out = GlobalAveragePooling3D()(input)
    out = Reshape(SE_shape)(out)
    out = Dense(nb_channel//ratio, activation = 'relu', kernel_initializer = 'he_normal', use_bias = False)(out)  # //表示相除并取整
    out = Dense(nb_channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(out)

    out = multiply([input, out])

    return out

def identity_block(x, nb_filters):
    k1, k2, k3 = nb_filters
    out = Convolution3D(k1, 1, strides=1, kernel_initializer='glorot_normal')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution3D(k2, 3, strides=1, kernel_initializer='glorot_normal', padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution3D(k3, 1, strides=1, kernel_initializer='glorot_normal')(out)
    out = BatchNormalization()(out)

    out = merge([out, x], mode='sum')
    #out = Add()([out, x])
    out = Activation('relu')(out)
    return out

def conv_block(x, nb_filters):
    k1, k2, k3 = nb_filters
    out = Convolution3D(k1, 1, strides=2, kernel_initializer='glorot_normal')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution3D(k2, 3, strides=1, kernel_initializer='glorot_normal', padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution3D(k3, 1, strides=1, kernel_initializer='glorot_normal')(out)
    out = BatchNormalization()(out)

    x1 = Convolution3D(k3, 1, strides=2, kernel_initializer='glorot_normal')(x)
    x1 = BatchNormalization()(x1)

    out = merge([out, x1], mode='sum')
    #out = Add()([out, x1])
    out = Activation('relu')(out)
    return out

def se_identity_block(x, nb_filters):
    k1, k2, k3 = nb_filters
    out = Convolution3D(k1, 1, strides=1, kernel_initializer='glorot_normal')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution3D(k2, 3, strides=1, kernel_initializer='glorot_normal', padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution3D(k3, 1, strides=1, kernel_initializer='glorot_normal')(out)
    out = BatchNormalization()(out)

    out = squeeze_excite_block3d(out)

    #out = Add()([out, x])
    out = merge([out, x], mode='sum')

    out = Activation('relu')(out)
    return out

def resnetttt(in_shape):
    inp = Input(shape=in_shape)
    out = ZeroPadding3D((3, 3, 3))(inp)
    out = Convolution3D(64, 5, strides=1)(out)
    print("input shape:", out.shape)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(out)
    print("shape:", out.shape)

    out = conv_block(out, [64, 64, 256])#[32, 32, 64]
    out = identity_block(out, [64, 64, 256])
    out = identity_block(out, [64, 64, 256])
    print("stage 1 shape:", out.shape)

    out = conv_block(out, [128, 128, 512])#[64, 64, 64]
    out = identity_block(out, [128, 128, 512])
    out = identity_block(out, [128, 128, 512])
    out = identity_block(out, [128, 128, 512])
    print("stage 2 shape:", out.shape)

    out = conv_block(out, [256, 256, 1024])
    out = se_identity_block(out, [256, 256, 1024])
    out = se_identity_block(out, [256, 256, 1024])
    out = se_identity_block(out, [256, 256, 1024])
    out = se_identity_block(out, [256, 256, 1024])
    out = se_identity_block(out, [256, 256, 1024])
    print("stage 3 shape:", out.shape)

    out = conv_block(out, [512, 512, 2048])
    out = se_identity_block(out, [512, 512, 2048])
    out = se_identity_block(out, [512, 512, 2048])
    print("stage 4 shape:", out.shape)

    out = GlobalAveragePooling3D(data_format = 'channels_last')(out)
    print("pooling shape:", out.shape)

    # out = Dense(1000, activation='softmax')(out)
    # out = Flatten()(out)
    # out = Dense(500, activation='relu')(out)
    # out = Dropout(rate=0.25)(out)

    out = Dense(150, activation='relu')(out)#250
    out = Dropout(rate=0.3)(out)
    print("dense shape:", out.shape)

    out = Dense(2, activation='softmax')(out)
    print("dense shape:", out.shape)

    model = Model(inp, out)

    model.compile(optimizer=SGD(lr=1e-6), loss='categorical_crossentropy', metrics=['acc'])
    return model

def alexnet_jn():
    inputs = Input(shape=(121, 145, 121, 1), name='input1')

    # 121x145x121
    conv1 = Conv3D(48, 7, strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1')(inputs)
    pool1 = MaxPooling3D(pool_size=2, padding='same', name='pool1')(conv1)
    bn1 = BatchNormalization(axis=1, name='batch_normalization_1')(pool1)
    print("conv1 shape:", conv1.shape)
    print("pool1 shape:", pool1.shape)
    # conv1 shape: (?, 121, 145, 121, 48)
    # pool1 shape: (?, 61, 73, 61, 48)
    conv2 = Conv3D(128, 5, strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2')(bn1)
    pool2 = MaxPooling3D(pool_size=2, padding='same', name='pool2')(conv2)
    bn2 = BatchNormalization(axis=1, name='batch_normalization_2')(pool2)
    print("conv2 shape:", conv2.shape)
    print("pool2 shape:", pool2.shape)
    # conv2 shape: (?, 61, 73, 61, 128)
    # pool2 shape: (?, 31, 37, 31, 128)
    conv3 = Conv3D(192, 3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3')(bn2)
    bn3 = BatchNormalization(axis=1, name='batch_normalization_3')(conv3)
    print("conv3 shape:", conv3.shape)
    # conv3 shape: (?, 31, 37, 31, 192)
    conv4 = Conv3D(192, 3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4')(bn3)
    bn4 = BatchNormalization(axis=1, name='batch_normalization_4')(conv4)
    print("conv4 shape:", conv4.shape)
    # conv3 shape: (?, 31, 37, 31, 192)
    conv5 = Conv3D(128, 3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='conv5')(bn4)
    pool3 = MaxPooling3D(pool_size=3, padding='same', name='pool3')(conv5)
    bn5 = BatchNormalization(axis=1, name='batch_normalization_5')(pool3)
    print("conv5 shape:", conv5.shape)
    print("pool3 shape:", pool3.shape)
    # conv5 shape: (?, 31, 37, 31, 128)
    # pool3 shape: (?, 11, 13, 11, 128)

    flatten1 = Flatten()(bn5)
    fc1 = Dense(500, activation='relu', name = 'fc1')(flatten1)
    # fc1_drop = Dropout(rate=0.25)(fc1)
    fc1_drop = Dropout(rate=0.25)(fc1)
    fc2 = Dense(250, activation='relu', name = 'fc2')(fc1_drop)
    # fc2_drop = Dropout(rate=0.25)(fc2)
    fc2_drop = Dropout(rate=0.25)(fc2)

    fc3 = Dense(2, name='fc3')(fc2_drop)
    output = Activation(activation='softmax')(fc3)

    model = Model(input=inputs, output=output)
    # model.compile(optimizer=SGD(lr=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.compile(optimizer=SGD(lr=1e-6), loss='categorical_crossentropy', metrics=['acc'])
    return model


