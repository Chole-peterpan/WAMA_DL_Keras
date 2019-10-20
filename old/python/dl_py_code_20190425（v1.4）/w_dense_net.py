from keras.models import Model
from keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D,  AveragePooling3D, GlobalAveragePooling3D, Dense, Dropout, Activation, multiply
from keras.layers.merge import concatenate
from w_other_block import squeeze_excite_block3d, sk_block3d








# nb_filter actually is growth_rate.这个模块就是每次输出增加的通道featuremap
def __conv_block(input, nb_filter, bottleneck=False,  dropout_rate=None, bias_allow=False, concat_axis=-1, bn_axis=-1, cbname='s'):

    x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5,name = cbname+'_bn')(input)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Conv3D(inter_channel, 1, strides=1, kernel_initializer='he_normal', use_bias=bias_allow,name = cbname+'botneck_conv')(x)
        x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5,name = cbname+'botneck_bn')(x)
        x = Activation('relu')(x)

    x = Conv3D(nb_filter, 3, strides=1, kernel_initializer='he_normal', padding='same', use_bias=bias_allow, name = cbname+'_conv')(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


# dense模块，增加通道数，增加变量nb_filter（这个变量主要是给过度模块使用的，dense block只有增加这个变量的功能，并没有用到这个变量），起到通道堆叠作用
def __dense_block(x, nb_layers, nb_filter, growth_rate, dbname, concat_axis=-1, bn_axis=-1, bottleneck=False, dropout_rate=None, grow_nb_filters=True):

    for i in range(nb_layers):
        #print('cb:', dbname+'_cb'+str(i))
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, concat_axis=concat_axis, bn_axis=bn_axis, cbname=dbname+'_cb'+str(i))
        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    return x, nb_filter


# 过渡模块，缩小张量size，不改变channel数，主要起到改变size的降采样作用
def __transition_block(input, nb_filter, tbname, compression=1.0, concat_axis=-1, bn_axis = -1, bias_allow=False):

    x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5, name = tbname+'_bn')(input)
    x = Activation('relu')(x)
    x = Conv3D(int(nb_filter * compression), 1, strides=1, kernel_initializer='he_normal', use_bias=bias_allow, name = tbname+'_conv')(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 1))(x)

    return x


# 构建densenet的最终函数，返回的就是densenet模型
# img_input:输入
# growth_rate=12：每过一个denseblock中的convblock，通道的增加数量
# reduction：denseblock通道衰减的比例
# dropout_rate:convblock的dropout_rate
# subsample_initial_block：是否提前进行个子采样
def dense_net(nb_layers, growth_rate=12, nb_filter=64, bottleneck=True, reduction=0.1, dropout_rate=None, subsample_initial_block=True, classes=2):

    inputs = Input(shape=(280, 280, 16, 1),name='input')
    print("0 :inputs shape:", inputs.shape)

    # 设定每个denseblock中convblock的数量:nb_layers = [3,3,3]

    concat_axis = -1  # 设定concat的轴（即叠加的轴）
    bn_axis = -1  # 设定BN的轴（即叠加的轴）
    nb_dense_block = nb_layers.__len__()  # nb_dense_block ：denseblock的数量，需要和nb_layers对应,nb_layers = [3,3,3],则nb_dense_block=3,即3个stage,每个stage有3个dense_block
    final_nb_layer = nb_layers[-1]
    compression = 1.0 - reduction  # denseblock的通道衰减率，即实际输出通道数=原输出通道数x通道衰减率

    # Initial convolution =======================================================================================
    if subsample_initial_block:
        initial_kernel = (7, 7, 7)
        initial_strides = (2, 2, 2)  # 这个地方需要跑一下实验看一下222好还是221好
    else:
        initial_kernel = (3, 3, 3)
        initial_strides = (1, 1, 1)

    x = Conv3D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same', strides=initial_strides, use_bias=False, name = 'init_conv')(inputs)
    x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5, name='init_bn')(x)
    x = Activation('relu')(x)


    if subsample_initial_block:
        x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)


    print("0 :Initial conv shape:", x.shape)
    # Initial convolution finished ================================================================================

    # Add dense blocks start  ==================================================================================
    for block_idx in range(nb_dense_block - 1):
        #print('db:','db'+str(block_idx))
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, concat_axis=concat_axis, bn_axis=bn_axis, bottleneck=bottleneck, dropout_rate=dropout_rate, grow_nb_filters=True, dbname = 'db'+str(block_idx))
        print(block_idx+1, ":dense_block shape:", x.shape)

        x = __transition_block(x, nb_filter, compression=compression, concat_axis=concat_axis, bias_allow=False, tbname = 'tb'+str(block_idx))
        print(block_idx+1, ":transition_block shape:", x.shape)

        nb_filter = int(nb_filter * compression)
    # Add dense blocks finish ==================================================================================

    # The last dense_block does not have a transition_block
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, concat_axis=concat_axis, bn_axis=bn_axis, bottleneck=bottleneck, dropout_rate=dropout_rate, grow_nb_filters=True, dbname = 'db_last')
    print(nb_dense_block, ":dense_block shape:", x.shape)

    x = BatchNormalization(axis=bn_axis, epsilon=1.1e-5, name='bn_last')(x)
    x = Activation('relu')(x)

    out = GlobalAveragePooling3D(data_format='channels_last')(x)
    print("GApooling shape:", out.shape)
    out_drop = Dropout(rate=0.3)(out)


    if classes == 1:
        output = Dense(classes, activation='sigmoid', name='fc1')(out_drop)
        print("predictions1 shape:", output.shape, 'activition:sigmoid')
    else:
        output = Dense(classes, activation='softmax', name='fc1')(out_drop)
        print("predictions2 shape:", output.shape, 'activition:softmax')



    #out = Dense(classes, name='fc1')(out_drop)
    #print("out shape:", out.shape)
    #output = Activation(activation='sigmoid')(out)

    model = Model(input=inputs, output=output)
    #mean_squared_logarithmic_error or binary_crossentropy
    #model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc] )

    return model


def se_dense_net(nb_layers, growth_rate=12, nb_filter=64, bottleneck=True, reduction=0.1, dropout_rate=None, subsample_initial_block=True,classes=2):

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
    out = Dense(classes, name='fc1')(out_drop)
    print("out shape:", out.shape)
    output = Activation(activation='sigmoid')(out)

    model = Model(input=inputs, output=output)
    #mean_squared_logarithmic_error or binary_crossentropy
    #model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc] )

    return model

# nestt = dense_net(nb_layers=[6,12,24,16],growth_rate=32, nb_filter=64, bottleneck=True, reduction=0.0, dropout_rate=None, subsample_initial_block=True, classes = 2)
# nestt = se_create_dense_net(nb_layers=[6,12,24,16],growth_rate=32, nb_filter=64, bottleneck=True, reduction=0.0, dropout_rate=None, subsample_initial_block=True)
# print(nestt.summary())


