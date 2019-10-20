from keras.models import *
from keras.layers import Input, merge, Conv3D, BatchNormalization, MaxPooling3D, GlobalAveragePooling3D,Dense,Lambda,Dropout,Activation,multiply,LeakyReLU
from keras.optimizers import *
from keras import backend as K
from keras.layers.core import Reshape
from keras.backend import int_shape
from keras.layers.merge import concatenate, add
from keras.backend import exp


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
    loss = (2 + 4 * a - 0.5 * b - 1 * c) * d + 0.2 * y_pred_f * d
    return loss


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


def identity_block(x, nb_filters, name, kernel_size=3, use_bias_flag=True):
    k1, k2, k3 = nb_filters
    convname1 = name + 'conv1'
    convname2 = name + 'conv2'
    convname3 = name + 'conv3'
    bnname1 = name + 'bn1'
    bnname2 = name + 'bn2'
    bnname3 = name + 'bn3'
    out = Conv3D(k1, 1, strides=1, kernel_initializer='he_normal', name = convname1, use_bias=use_bias_flag)(x)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname1)(out)
    out = Activation('relu')(out)


    out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2, use_bias=use_bias_flag)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname2)(out)
    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name = convname3, use_bias=use_bias_flag)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname3)(out)

    out = add([out, x])
    # out = merge([out, x], mode='sum')
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

    out = add([out, x])
    # out = merge([out, x], mode='sum')
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def conv_block(x, nb_filters, name, kernel_size=3, use_bias_flag=True):
    k1, k2, k3 = nb_filters
    convname1 = name + 'conv1'
    convname2 = name + 'conv2'
    convname3 = name + 'conv3'
    convname4 = name + 'conv4'
    bnname1 = name + 'bn1'
    bnname2 = name + 'bn2'
    bnname3 = name + 'bn3'
    bnname4 = name + 'bn4'
    out = Conv3D(k1, 2, strides=2, kernel_initializer='he_normal', name=convname1, use_bias=use_bias_flag)(x)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname1)(out)
    out = Activation('relu')(out)

    out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name=convname2, use_bias=use_bias_flag)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6, name = bnname2)(out)
    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name=convname3, use_bias=use_bias_flag)(out)
    out = BatchNormalization(axis = -1, epsilon = 1e-6,  name = bnname3)(out)

    x1 = Conv3D(k3, 2, strides=2, kernel_initializer='he_normal', name = convname4, use_bias=use_bias_flag)(x)
    x1 = BatchNormalization(axis = -1, epsilon = 1e-6, name = bnname4)(x1)

    out = add([out, x1])
    # out = merge([out, x1], mode='sum')
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def resnet(use_bias_flag=True):
    '''
    :param use_bias_flag: 是否使用偏置，包括卷积层与全连接层
    :param bn_flag: 是否使用bn层，全网络范围
    :return:resnet模型
    '''

    inputs = Input(shape=(280, 280, 16, 1), name='input1')
    # 256*256*128
    print("input shape:", inputs.shape)  # (?, 140, 140, 16, 64)
    out = Conv3D(64, 7, strides=(2, 2, 1), padding='same', kernel_initializer='he_normal', use_bias=use_bias_flag, name='conv1')(inputs)
    print("conv0 shape:", out.shape)#(?, 140, 140, 16, 64)
    out = BatchNormalization(axis=-1, epsilon=1e-6, name='bn1')(out)
    out = Activation('relu')(out)
    out = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same')(out)
    print("pooling1 shape:", out.shape)#(?, 70, 70, 16, 64)

    out = conv_block(out, [64, 64, 256], name='L1_block1', use_bias_flag=use_bias_flag)
    print("conv1 shape:", out.shape)
    out = identity_block(out, [64, 64, 256], name='L1_block2', use_bias_flag=use_bias_flag)
    out = identity_block(out, [64, 64, 256], name='L1_block3', use_bias_flag=use_bias_flag)

    out = conv_block(out, [128, 128, 512], name='L2_block1', use_bias_flag=use_bias_flag)
    print("conv2 shape:", out.shape)
    out = identity_block(out, [128, 128, 512], name='L2_block2', use_bias_flag=use_bias_flag)
    out = identity_block(out, [128, 128, 512], name='L2_block3', use_bias_flag=use_bias_flag)
    out = identity_block(out, [128, 128, 512], name='L2_block4', use_bias_flag=use_bias_flag)


    out = conv_block(out, [256, 256, 1024], name = 'L3_block1', use_bias_flag=use_bias_flag)
    print("conv3 shape:", out.shape)
    out = identity_block(out, [256, 256, 1024], name='L3_block2', use_bias_flag=use_bias_flag)
    out = identity_block(out, [256, 256, 1024], name='L3_block3', use_bias_flag=use_bias_flag)
    out = identity_block(out, [256, 256, 1024], name='L3_block4', use_bias_flag=use_bias_flag)
    out = identity_block(out, [256, 256, 1024], name='L3_block5', use_bias_flag=use_bias_flag)
    out = identity_block(out, [256, 256, 1024], name='L3_block6', use_bias_flag=use_bias_flag)

    out = conv_block(out, [512, 512, 2048], name='L4_block1', use_bias_flag=use_bias_flag)
    print("conv4 shape:", out.shape)
    out = identity_block(out, [512, 512, 2048], name='L4_block2', use_bias_flag=use_bias_flag)
    out = identity_block(out, [512, 512, 2048], name='L4_block3', use_bias_flag=use_bias_flag)

    out = GlobalAveragePooling3D(data_format='channels_last')(out)
    print("Gpooling shape:", out.shape)
    out_drop = Dropout(rate=0.3)(out)
    out = Dense(1, name='fc1', use_bias=use_bias_flag)(out_drop)
    print("out shape:", out.shape)
    output = Activation(activation='sigmoid')(out)

    model = Model(input=inputs, output=output)
    #mean_squared_logarithmic_error or binary_crossentropy
    #model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    return model

def se_resnet():
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
    #model.compile(optimizer=SGD(lr = 1e-6, momentum = 0.9), loss = EuiLoss, metrics= [y_t, y_pre, Acc] )
    return model



# multi_input_net
# 可有多输入的resnet
# input1:动脉期 a
# input2:门脉期 v
def multiinput_resnet():
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
    #model.compile(optimizer=SGD(lr = 1e-6, momentum = 0.9), loss = EuiLoss, metrics= [y_t, y_pre, Acc] )

    print('im multi_input_ClassNet')
    return model

# 多输入resnet的leaky relu 版本 =========================================================================
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

def multiinput_resnet_lk():


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
    #model.compile(optimizer=SGD(lr = 1e-6, momentum = 0.9), loss = EuiLoss, metrics= [y_t, y_pre, Acc] )

    print('im multi_input_ClassNet_lk')
    return model

# p relu 版本 =========================================================================




# 没有bn的resnet普通版本
def identity_block_nobn(x, nb_filters, name, kernel_size=3, use_bias_flag=True):
    k1, k2, k3 = nb_filters
    convname1 = name + 'conv1'
    convname2 = name + 'conv2'
    convname3 = name + 'conv3'
    bnname1 = name + 'bn1'
    bnname2 = name + 'bn2'
    bnname3 = name + 'bn3'
    out = Conv3D(k1, 1, strides=1, kernel_initializer='he_normal', name = convname1, use_bias=use_bias_flag)(x)

    out = Activation('relu')(out)


    out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name = convname2, use_bias=use_bias_flag)(out)

    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name = convname3, use_bias=use_bias_flag)(out)


    out = add([out, x])
    # out = merge([out, x], mode='sum')

    out = Activation('relu')(out)
    return out

def conv_block_nobn(x, nb_filters, name, kernel_size=3, use_bias_flag=True):
    k1, k2, k3 = nb_filters
    convname1 = name + 'conv1'
    convname2 = name + 'conv2'
    convname3 = name + 'conv3'
    convname4 = name + 'conv4'
    bnname1 = name + 'bn1'
    bnname2 = name + 'bn2'
    bnname3 = name + 'bn3'
    bnname4 = name + 'bn4'
    out = Conv3D(k1, 2, strides=2, kernel_initializer='he_normal', name=convname1, use_bias=use_bias_flag)(x)

    out = Activation('relu')(out)

    out = Conv3D(k2, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name=convname2, use_bias=use_bias_flag)(out)

    out = Activation('relu')(out)

    out = Conv3D(k3, 1, strides=1, kernel_initializer='he_normal', name=convname3, use_bias=use_bias_flag)(out)


    x1 = Conv3D(k3, 2, strides=2, kernel_initializer='he_normal', name = convname4, use_bias=use_bias_flag)(x)


    out = add([out, x1])
    # out = merge([out, x1], mode='sum')

    out = Activation('relu')(out)
    return out


def resnet_nobn(use_bias_flag=True):
    '''
    :param use_bias_flag: 是否使用偏置，包括卷积层与全连接层
    :param bn_flag: 是否使用bn层，全网络范围
    :return:resnet模型
    '''

    inputs = Input(shape=(280, 280, 16, 1), name='input1')
    # 256*256*128
    print("input shape:", inputs.shape)  # (?, 140, 140, 16, 64)
    out = Conv3D(64, 7, strides=(2, 2, 1), padding='same', kernel_initializer='he_normal', use_bias=use_bias_flag, name='conv1')(inputs)
    print("conv0 shape:", out.shape)#(?, 140, 140, 16, 64)

    out = Activation('relu')(out)
    out = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same')(out)
    print("pooling1 shape:", out.shape)#(?, 70, 70, 16, 64)

    out = conv_block_nobn(out, [64, 64, 256], name='L1_block1', use_bias_flag=use_bias_flag)
    print("conv1 shape:", out.shape)
    out = identity_block_nobn(out, [64, 64, 256], name='L1_block2', use_bias_flag=use_bias_flag)
    out = identity_block_nobn(out, [64, 64, 256], name='L1_block3', use_bias_flag=use_bias_flag)

    out = conv_block_nobn(out, [128, 128, 512], name='L2_block1', use_bias_flag=use_bias_flag)
    print("conv2 shape:", out.shape)
    out = identity_block_nobn(out, [128, 128, 512], name='L2_block2', use_bias_flag=use_bias_flag)
    out = identity_block_nobn(out, [128, 128, 512], name='L2_block3', use_bias_flag=use_bias_flag)
    out = identity_block_nobn(out, [128, 128, 512], name='L2_block4', use_bias_flag=use_bias_flag)


    out = conv_block_nobn(out, [256, 256, 1024], name = 'L3_block1', use_bias_flag=use_bias_flag)
    print("conv3 shape:", out.shape)
    out = identity_block_nobn(out, [256, 256, 1024], name='L3_block2', use_bias_flag=use_bias_flag)
    out = identity_block_nobn(out, [256, 256, 1024], name='L3_block3', use_bias_flag=use_bias_flag)
    out = identity_block_nobn(out, [256, 256, 1024], name='L3_block4', use_bias_flag=use_bias_flag)
    out = identity_block_nobn(out, [256, 256, 1024], name='L3_block5', use_bias_flag=use_bias_flag)
    out = identity_block_nobn(out, [256, 256, 1024], name='L3_block6', use_bias_flag=use_bias_flag)

    out = conv_block_nobn(out, [512, 512, 2048], name='L4_block1', use_bias_flag=use_bias_flag)
    print("conv4 shape:", out.shape)
    out = identity_block_nobn(out, [512, 512, 2048], name='L4_block2', use_bias_flag=use_bias_flag)
    out = identity_block_nobn(out, [512, 512, 2048], name='L4_block3', use_bias_flag=use_bias_flag)

    out = GlobalAveragePooling3D(data_format='channels_last')(out)
    print("Gpooling shape:", out.shape)
    out_drop = Dropout(rate=0.3)(out)
    out = Dense(1, name='fc1', use_bias=use_bias_flag)(out_drop)
    print("out shape:", out.shape)
    output = Activation(activation='sigmoid')(out)

    model = Model(input=inputs, output=output)
    #mean_squared_logarithmic_error or binary_crossentropy
    #model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    return model


