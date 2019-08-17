from keras.layers import Conv3D, BatchNormalization, GlobalAveragePooling3D, Dense, Lambda, Activation, multiply
from keras.optimizers import *
from keras.layers.core import Reshape
from keras.backend import int_shape
from keras.layers.merge import concatenate, add
from keras.backend import exp


# ----------------------------------------------------------------------------------
# 分组卷积函数 3D版本
# input的通道数需要等于grouped_channels*cardinality，也就是说运行函数之前需要计算一下这两个参数，而不是自动计算。
def _grouped_convolution_block_3D(input, grouped_channels, cardinality, strides, kernel_size):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters  每组包括的卷积核数量
        cardinality: cardinality factor describing the number of groups 组数，分组卷积一共分几组的组数
        strides: performs strided convolution for downscaling if > 1
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    group_list = []

    # 如果只需要分一组进行卷积，那就和没分一样
    # 直接局卷个积return就完了
    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv3D(grouped_channels, kernel_size, padding='same', use_bias=False, strides=strides, kernel_initializer='he_normal')(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    # 如果需要分多个组卷积，则标准分组卷积过程如下：
    # 先使用lambda层沿通道轴分割出各个组，之后再卷积，最后再沿通道轴concatenate合并
    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, :, c * grouped_channels:(c + 1) * grouped_channels] if K.image_data_format() == 'channels_last' else lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :, :])(input)
        x = Conv3D(grouped_channels, kernel_size, padding='same', use_bias=False, strides=strides, kernel_initializer='he_normal')(x)
        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    group_merge = BatchNormalization(axis=channel_axis)(group_merge)
    group_merge = Activation('relu')(group_merge)
    return group_merge


# 这个se模块好像没有考虑到batch？？
def squeeze_excite_block3d(input, ratio=2):
    nb_channel = int_shape(input)[-1]
    se_shape = (1, 1, 1, nb_channel)

    out = GlobalAveragePooling3D()(input)
    out = Reshape(se_shape)(out)
    out = Dense(nb_channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(out)  # //表示相除并取整
    out = Dense(nb_channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(out)

    out = multiply([input, out])

    return out


# 这个sk模块好像也没考虑到batch？？
def sk_block(input, ratio=2):
    nb_channel = int_shape(input)[-1]
    u1 = Conv3D(nb_channel, 3, strides=1, padding='same', kernel_initializer='he_normal')(input)
    u2 = Conv3D(nb_channel, 5, strides=1, padding='same', kernel_initializer='he_normal')(input)

    u = add([u1, u2])
    se_shape = (1, 1, 1, nb_channel)
    weight = GlobalAveragePooling3D()(u)
    weight = Reshape(se_shape)(weight)
    weight = Dense(nb_channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(weight)  # //表示相除并向下取整
    weight_u1 = Dense(nb_channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(weight)
    weight_u2 = Dense(nb_channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(weight)

    # 实现softmax---------------------------------------------------------------
    weight_u1 = exp(weight_u1)  # 取指数
    weight_u2 = exp(weight_u2)  # 取指数
    weight_mother = add([weight_u1, weight_u2])  # 求和
    weight_mother = Lambda(lambda z: 1/z)(weight_mother)  # 取倒数

    weight_u1 = multiply([weight_u1, weight_mother])  # 乘以倒数
    weight_u2 = multiply([weight_u2, weight_mother])  # 乘以倒数

    # 赋权重-------------------------------------------------------
    out1 = multiply([u1, weight_u1])
    out2 = multiply([u2, weight_u2])

    # 最终输出
    final_output = add([out1, out2])
    return final_output


# 这个sk模块好像也没考虑到batch？？
# 使用分组卷积的版本
def sk_block_g(input, ratio=2):
    nb_channel = int_shape(input)[-1]
    cardinality = 32  # 分32组
    grouped_channels = nb_channel // cardinality  # 除法运算并向下取整
    u1 = _grouped_convolution_block_3D(input, grouped_channels, cardinality, strides=1, kernel_size=3)
    u2 = _grouped_convolution_block_3D(input, grouped_channels, cardinality, strides=1, kernel_size=5)

    u = add([u1, u2])
    se_shape = (1, 1, 1, nb_channel)
    weight = GlobalAveragePooling3D()(u)
    weight = Reshape(se_shape)(weight)
    weight = Dense(nb_channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(weight)  # //表示相除并向下取整
    weight_u1 = Dense(nb_channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(weight)
    weight_u2 = Dense(nb_channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(weight)

    # 实现softmax---------------------------------------------------------------
    weight_u1 = exp(weight_u1)  # 取指数
    weight_u2 = exp(weight_u2)  # 取指数
    weight_mother = add([weight_u1, weight_u2])  # 求和
    weight_mother = Lambda(lambda z: 1/z)(weight_mother)  # 取倒数

    weight_u1 = multiply([weight_u1, weight_mother])  # 乘以倒数
    weight_u2 = multiply([weight_u2, weight_mother])  # 乘以倒数

    # 赋权重-------------------------------------------------------
    out1 = multiply([u1, weight_u1])
    out2 = multiply([u2, weight_u2])

    # 最终输出
    final_output = add([out1, out2])
    return final_output










