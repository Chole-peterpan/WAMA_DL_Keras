from keras.layers import merge, multiply
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D, AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, merge, Conv3D, BatchNormalization, Conv3DTranspose, UpSampling3D, MaxPooling3D, \
     AveragePooling3D, GlobalAveragePooling3D, Dense, Flatten, Lambda, Dropout, Activation, ZeroPadding3D
from keras.layers.merge import concatenate
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.backend import int_shape
from keras.layers.core import Reshape
# step1: import model finished
from function import sigmoid_y, pause, verify_on_model, test_on_model
from net import resnetttt, alexnet_jn  # use as :Resnetttt(tuple(data_input_shape+[1]))
from Net_new import ClassNet
from sklearn import cross_validation, metrics


def Acc(y_true, y_pred):
    y_pred_r = K.round(y_pred)
    return K.equal(y_pred_r, y_true)


def y_t(y_true, y_pred):
    return y_true


def y_pre(y_true, y_pred):
    return y_pred


def EuiLoss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
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


# nb_filter actually is growth_rate.
def __conv_block(input, nb_filter, bottleneck=False,  dropout_rate=None, bias_allow=False, concat_axis=-1, bn_axis=-1):

    x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Conv3D(inter_channel, 1, strides=1, kernel_initializer='he_normal', use_bias=bias_allow)(x)
        x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv3D(nb_filter, 3, strides=1, kernel_initializer='he_normal', padding='same', use_bias=bias_allow)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


# dense模块，增加通道数，增加变量nb_filter，起到通道堆叠作用
def __dense_block(x, nb_layers, nb_filter, growth_rate, concat_axis=-1, bn_axis=-1, bottleneck=False, dropout_rate=None, grow_nb_filters=True):

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, concat_axis=concat_axis, bn_axis=bn_axis)
        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    return x, nb_filter


# 过渡模块，较小张量size，不改变channel数
def __transition_block(input, nb_filter, compression=1.0, concat_axis=-1, bn_axis = -1, bias_allow=False):

    x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv3D(int(nb_filter * compression), 1, strides=1, kernel_initializer='he_normal', use_bias=bias_allow)(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 1))(x)

    return x


# 构建densenet的最终函数，返回的就是densenet模型
# img_input:输入
# growth_rate=12：每过一个denseblock中的convblock，通道的增加数量
# reduction：denseblock通道衰减的比例
# dropout_rate:convblock的dropout_rate
# subsample_initial_block：是否提前进行个子采样
def create_dense_net(nb_layers, growth_rate=12, nb_filter=64, bottleneck=True, reduction=0.1, dropout_rate=None,
                       subsample_initial_block=True):

    inputs = Input(shape=(280, 280, 16, 1))
    print("0 :inputs shape:", inputs.shape)

    # 设定每个denseblock中convblock的数量:nb_layers = [3,3,3]

    concat_axis = -1  # 设定concat的轴（即叠加的轴）
    bn_axis = -1  # 设定BN的轴（即叠加的轴）
    nb_dense_block = nb_layers.__len__()  # nb_dense_block ：denseblock的数量，需要和nb_layers对应
    final_nb_layer = nb_layers[-1]
    compression = 1.0 - reduction  # denseblock的通道衰减率，即实际输出通道数=原输出通道数x通道衰减率

    # Initial convolution =======================================================================================
    if subsample_initial_block:
        initial_kernel = (7, 7, 7)
        initial_strides = (2, 2, 1)
    else:
        initial_kernel = (3, 3, 3)
        initial_strides = (1, 1, 1)

    x = Conv3D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False)(inputs)

    if subsample_initial_block:
        x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same')(x)

    print("0 :Initial conv shape:", x.shape)
    # Initial convolution finished ================================================================================

    # Add dense blocks start  ==================================================================================
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, concat_axis=concat_axis,
                                     bn_axis=bn_axis, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, grow_nb_filters=True)
        print(block_idx+1, ":dense_block shape:", x.shape)

        x = __transition_block(x, nb_filter, compression=compression, concat_axis=concat_axis, bias_allow=False)
        print(block_idx+1, ":transition_block shape:", x.shape)

        nb_filter = int(nb_filter * compression)
    # Add dense blocks finish ==================================================================================

    # The last dense_block does not have a transition_block
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, concat_axis=concat_axis, bn_axis=bn_axis,
                                 bottleneck=bottleneck, dropout_rate=dropout_rate, grow_nb_filters=True)
    print(nb_dense_block, ":dense_block shape:", x.shape)

    x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    out = GlobalAveragePooling3D(data_format='channels_last')(x)
    print("GApooling shape:", out.shape)
    out_drop = Dropout(rate=0.3)(out)
    out = Dense(1, name='fc1')(out_drop)
    print("out shape:", out.shape)
    output = Activation(activation='sigmoid')(out)

    model = Model(input=inputs, output=output)
    #mean_squared_logarithmic_error or binary_crossentropy
    model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc] )

    return model


def se_create_dense_net(nb_layers, growth_rate=12, nb_filter=64, bottleneck=True, reduction=0.1, dropout_rate=None,
                       subsample_initial_block=True):

    inputs = Input(shape=(280, 280, 16, 1))
    print("0 :inputs shape:", inputs.shape)

    # 设定每个denseblock中convblock的数量:nb_layers = [3,3,3]

    concat_axis = -1  # 设定concat的轴（即叠加的轴）
    bn_axis = -1  # 设定BN的轴（即叠加的轴）
    nb_dense_block = nb_layers.__len__()  # nb_dense_block ：denseblock的数量，需要和nb_layers对应
    final_nb_layer = nb_layers[-1]
    compression = 1.0 - reduction  # denseblock的通道衰减率，即实际输出通道数=原输出通道数x通道衰减率

    # Initial convolution =======================================================================================
    if subsample_initial_block:
        initial_kernel = (7, 7, 7)
        initial_strides = (2, 2, 1)
    else:
        initial_kernel = (3, 3, 3)
        initial_strides = (1, 1, 1)

    x = Conv3D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False)(inputs)

    if subsample_initial_block:
        x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same')(x)

    print("0 :Initial conv shape:", x.shape)
    # Initial convolution finished ================================================================================

    # Add dense blocks start  ==================================================================================
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, concat_axis=concat_axis,
                                     bn_axis=bn_axis, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, grow_nb_filters=True)
        print(block_idx+1, ":dense_block shape:", x.shape)

        x = __transition_block(x, nb_filter, compression=compression, concat_axis=concat_axis, bias_allow=False)
        print(block_idx+1, ":transition_block shape:", x.shape)

        x = squeeze_excite_block3d(x)
        print(block_idx + 1, ":se_block_out shape:", x.shape)

        nb_filter = int(nb_filter * compression)
    # Add dense blocks finish ==================================================================================

    # The last dense_block does not have a transition_block
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, concat_axis=concat_axis, bn_axis=bn_axis,
                                 bottleneck=bottleneck, dropout_rate=dropout_rate, grow_nb_filters=True)
    print(nb_dense_block, ":dense_block shape:", x.shape)

    x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    out = GlobalAveragePooling3D(data_format='channels_last')(x)
    print("GApooling shape:", out.shape)
    out_drop = Dropout(rate=0.3)(out)
    out = Dense(1, name='fc1')(out_drop)
    print("out shape:", out.shape)
    output = Activation(activation='sigmoid')(out)

    model = Model(input=inputs, output=output)
    #mean_squared_logarithmic_error or binary_crossentropy
    model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc] )

    return model

# nestt = create_dense_net(nb_layers=[6,12,24,16],growth_rate=32, nb_filter=64, bottleneck=True, reduction=0.0, dropout_rate=None, subsample_initial_block=True)
# nestt = se_create_dense_net(nb_layers=[6,12,24,16],growth_rate=32, nb_filter=64, bottleneck=True, reduction=0.0, dropout_rate=None, subsample_initial_block=True)
# print(nestt.summary())


