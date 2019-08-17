from keras.models import *
from keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, GlobalAveragePooling3D, Dense, Lambda, Dropout, Activation, multiply
from keras.optimizers import *
from keras import backend as K
from keras.layers.core import Reshape
from keras.backend import int_shape
from keras.layers.merge import concatenate, add





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


# ----------------------------------------------------------------------------------
# 分组卷积函数
# input的通道数需要等于grouped_channels*cardinality，也就是说运行函数之前需要计算一下这两个参数，而不是自动计算。
def _grouped_convolution_block_3D(input, grouped_channels, cardinality, strides):
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
        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=strides, kernel_initializer='he_normal')(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    # 如果需要分多个组卷积，则标准分组卷积过程如下：
    # 先使用lambda层沿通道轴分割出各个组，之后再卷积，最后再沿通道轴concatenate合并
    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, :, c * grouped_channels:(c + 1) * grouped_channels] if K.image_data_format() == 'channels_last' else lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :, :])(input)
        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=strides, kernel_initializer='he_normal')(x)
        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    #group_merge = BatchNormalization(axis=channel_axis)(group_merge)
    #group_merge = Activation('relu')(group_merge)
    return group_merge


def identity_block(x, nb_filters, name, cardinality=32):
    k1, k2, k3 = nb_filters
    convname1 = name + 'conv1'
    convname2 = name + 'conv2'
    convname3 = name + 'conv3'
    bnname1 = name + 'bn1'
    bnname2 = name + 'bn2'
    bnname3 = name + 'bn3'
    out = Conv3D(k1, 1, strides=1, kernel_initializer='he_normal', name=convname1)(x)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname1)(out)
    out = Activation('relu')(out)

    grouped_channels = k2 // cardinality  # 除法运算并向下取整
    out = _grouped_convolution_block_3D(out, grouped_channels, cardinality, strides=1)
    #out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname2)(out)
    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name = convname3)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname3)(out)

    out = add([out, x])
    #out = merge([out, x], mode='sum')
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def se_identity_block(x, nb_filters, name, cardinality=32):
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

    grouped_channels = k2 // cardinality  # 除法运算并向下取整
    out = _grouped_convolution_block_3D(out, grouped_channels, cardinality, strides=1)
    #out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2)(out)
    out = BatchNormalization(axis=-1, epsilon=1e-6,  name = bnname2)(out)
    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name = convname3)(out)
    out = BatchNormalization(axis=-1, epsilon=1e-6,  name=bnname3)(out)

    out = squeeze_excite_block3d(out)

    out = add([out, x])
    #out = merge([out, x], mode='sum')
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


#  先尺寸下降，再分组卷积
def conv_block(x, nb_filters, name, cardinality=32):
    k1, k2, k3 = nb_filters
    convname1 = name + 'conv1'
    convname2 = name + 'conv2'
    convname3 = name + 'conv3'
    convname4 = name + 'conv4'
    bnname1 = name + 'bn1'
    bnname2 = name + 'bn2'
    bnname3 = name + 'bn3'
    bnname4 = name + 'bn4'
    out = Conv3D(k1, 2, strides=2, kernel_initializer='he_normal', name=convname1)(x)
    out = BatchNormalization(axis=-1, epsilon=1e-6,  name=bnname1)(out)
    out = Activation('relu')(out)

    grouped_channels = k2 // cardinality  # 除法运算并向下取整
    out = _grouped_convolution_block_3D(out, grouped_channels, cardinality, strides=1)
    #out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2)(out)
    out = BatchNormalization(axis=-1, epsilon=1e-6, name=bnname2)(out)
    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name = convname3)(out)
    out = BatchNormalization(axis=-1, epsilon=1e-6,  name=bnname3)(out)

    x1 = Conv3D(k3, 2, strides=2, kernel_initializer='he_normal', name=convname4)(x)
    x1 = BatchNormalization(axis=-1, epsilon=1e-6, name=bnname4)(x1)

    out = add([out, x1])
    #out = merge([out, x1], mode='sum')
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


#  另外一个版本，分组卷积时候尺寸下降，而不是之前的先下降尺寸再分组
def conv_block_or(x, nb_filters, name, cardinality=32):
    k1, k2, k3 = nb_filters
    convname1 = name + 'conv1'
    convname2 = name + 'conv2'
    convname3 = name + 'conv3'
    convname4 = name + 'conv4'
    bnname1 = name + 'bn1'
    bnname2 = name + 'bn2'
    bnname3 = name + 'bn3'
    bnname4 = name + 'bn4'
    out = Conv3D(k1, 1, strides=1, kernel_initializer='he_normal', name=convname1)(x)
    out = BatchNormalization(axis=-1, epsilon=1e-6,  name=bnname1)(out)
    out = Activation('relu')(out)

    grouped_channels = k2 // cardinality  # 除法运算并向下取整
    out = _grouped_convolution_block_3D(out, grouped_channels, cardinality, strides=2)
    #out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2)(out)
    out = BatchNormalization(axis=-1, epsilon=1e-6, name=bnname2)(out)
    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name=convname3)(out)
    out = BatchNormalization(axis=-1, epsilon=1e-6,  name=bnname3)(out)

    x1 = Conv3D(k3, 3, strides=2, padding='same', kernel_initializer='he_normal', name=convname4)(x)
    x1 = BatchNormalization(axis=-1, epsilon=1e-6, name=bnname4)(x1)

    out = add([out, x1])
    #out = merge([out, x1], mode='sum')
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


#先尺寸下降，再分组卷积
def resnext(classes=2):
    inputs = Input(shape=(280, 280, 16, 1), name='input1')
    # 256*256*128
    print("input shape:", inputs.shape)  # (?, 140, 140, 16, 64)
    out = Conv3D(64, 7, strides=(2, 2, 1), padding='same', kernel_initializer='he_normal', use_bias=False, name='conv1')(inputs)
    print("conv0 shape:", out.shape)#(?, 140, 140, 16, 64)
    out = BatchNormalization(axis = -1, epsilon=1e-6, name='bn1')(out)
    out = Activation('relu')(out)
    out = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same')(out)
    print("pooling1 shape:", out.shape)#(?, 70, 70, 16, 64)

    out = conv_block(out, [64, 64, 256], name='L1_block1')  # 一定是[n,n,Xn]这个形式（因为有分组卷积），X可取任意值，且输出通道数为Xn,另外n最好是32的整数（因为分组卷积是分32组的，当然这个可以自己改）
    print("conv1 shape:", out.shape)
    out = identity_block(out, [64, 64, 256], name='L1_block2')  # 一定是[n,n,a]的形式，a一定要等于上一个conv_block或identity_block的输出通道，identity_block的输入输出通道相同。

    out = identity_block(out, [64, 64, 256], name='L1_block3')

    out = conv_block(out, [128, 128, 512], name='L2_block1')
    print("conv2 shape:", out.shape)
    out = identity_block(out, [128, 128, 512], name='L2_block2')

    out = identity_block(out, [128, 128, 512], name='L2_block3')

    out = identity_block(out, [128, 128, 512], name='L2_block4')

    out = conv_block(out, [256, 256, 1024], name='L3_block1')
    print("conv3 shape:", out.shape)
    out = identity_block(out, [256, 256, 1024], name='L3_block2')
    out = identity_block(out, [256, 256, 1024], name='L3_block3')
    out = identity_block(out, [256, 256, 1024], name='L3_block4')
    out = identity_block(out, [256, 256, 1024], name='L3_block5')
    out = identity_block(out, [256, 256, 1024], name='L3_block6')

    out = conv_block(out, [512, 512, 2048], name='L4_block1')
    print("conv4 shape:", out.shape)
    out = identity_block(out, [512, 512, 2048], name='L4_block2')
    out = identity_block(out, [512, 512, 2048], name='L4_block3')

    out = GlobalAveragePooling3D(data_format = 'channels_last')(out)
    print("Gpooling shape:", out.shape)
    out_drop = Dropout(rate=0.3)(out)
    out = Dense(classes, name = 'fc1')(out_drop)
    print("out shape:", out.shape)
    #out = Dense(1, name = 'fc1')(out)
    output = Activation(activation = 'sigmoid')(out)

    model = Model(input = inputs, output = output)
    #mean_squared_logarithmic_error or binary_crossentropy
    #model.compile(optimizer=SGD(lr = 1e-6, momentum = 0.9), loss = EuiLoss, metrics= [y_t, y_pre, Acc] )
    return model


#  再分组卷积的时候尺寸下降
def resnext_or(classes=2):
    inputs = Input(shape=(280, 280, 16, 1), name='input1')
    # 256*256*128
    print("input shape:", inputs.shape)  # (?, 140, 140, 16, 64)
    out = Conv3D(64, 7, strides=(2, 2, 1), padding='same', kernel_initializer='he_normal', use_bias=False, name='conv1')(inputs)
    print("conv0 shape:", out.shape)#(?, 140, 140, 16, 64)
    out = BatchNormalization(axis = -1, epsilon=1e-6, name='bn1')(out)
    out = Activation('relu')(out)
    out = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same')(out)
    print("pooling1 shape:", out.shape)#(?, 70, 70, 16, 64)

    out = conv_block_or(out, [64, 64, 256], name='L1_block1')  # 一定是[n,n,Xn]这个形式（因为有分组卷积），X可取任意值，且输出通道数为Xn,另外n最好是32的整数（因为分组卷积是分32组的，当然这个可以自己改）
    print("conv1 shape:", out.shape)
    out = identity_block(out, [64, 64, 256], name='L1_block2')  # 一定是[n,n,a]的形式，a一定要等于上一个conv_block或identity_block的输出通道，identity_block的输入输出通道相同。
    out = identity_block(out, [64, 64, 256], name='L1_block3')

    out = conv_block_or(out, [128, 128, 512], name='L2_block1')
    print("conv2 shape:", out.shape)
    out = identity_block(out, [128, 128, 512], name='L2_block2')
    out = identity_block(out, [128, 128, 512], name='L2_block3')
    out = identity_block(out, [128, 128, 512], name='L2_block4')

    out = conv_block_or(out, [256, 256, 1024], name='L3_block1')
    print("conv3 shape:", out.shape)
    out = identity_block(out, [256, 256, 1024], name='L3_block2')
    out = identity_block(out, [256, 256, 1024], name='L3_block3')
    out = identity_block(out, [256, 256, 1024], name='L3_block4')
    out = identity_block(out, [256, 256, 1024], name='L3_block5')
    out = identity_block(out, [256, 256, 1024], name='L3_block6')

    out = conv_block_or(out, [512, 512, 2048], name='L4_block1')
    print("conv4 shape:", out.shape)
    out = identity_block(out, [512, 512, 2048], name='L4_block2')
    out = identity_block(out, [512, 512, 2048], name='L4_block3')

    out = GlobalAveragePooling3D(data_format = 'channels_last')(out)
    print("Gpooling shape:", out.shape)
    out_drop = Dropout(rate=0.3)(out)
    out = Dense(classes, name = 'fc1')(out_drop)
    print("out shape:", out.shape)
    #out = Dense(1, name = 'fc1')(out)
    output = Activation(activation = 'sigmoid')(out)

    model = Model(input = inputs, output = output)
    #mean_squared_logarithmic_error or binary_crossentropy
    #model.compile(optimizer=SGD(lr = 1e-6, momentum = 0.9), loss = EuiLoss, metrics= [y_t, y_pre, Acc] )
    return model


#先尺寸下降，再分组卷积
def se_resnext(classes=2):
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
    out = Dense(classes, name = 'fc1')(out_drop)
    print("out shape:", out.shape)
    #out = Dense(1, name = 'fc1')(out)
    output = Activation(activation = 'sigmoid')(out)

    model = Model(input = inputs, output = output)
    #mean_squared_logarithmic_error or binary_crossentropy
    #model.compile(optimizer=SGD(lr = 1e-6, momentum = 0.9), loss = EuiLoss, metrics= [y_t, y_pre, Acc] )
    return model


# multi_input_net
# 可有多输入的resnext,先尺寸下降，再分组卷积
# input1:动脉期 a
# input2:门脉期 v
def multiinput_resnext(classes=2):
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
    out = Dense(classes, name = 'fc1')(out_drop)
    print("out shape:", out.shape)
    output = Activation(activation = 'sigmoid')(out)

    model = Model(input = [inputs_1, inputs_2], output = output)
    #mean_squared_logarithmic_error or binary_crossentropy
    model.compile(optimizer=SGD(lr = 1e-6, momentum = 0.9), loss = EuiLoss, metrics= [y_t, y_pre, Acc] )

    print('im multi_input_ClassNet')
    return model










