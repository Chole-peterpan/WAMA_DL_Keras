from keras.layers import Lambda, BatchNormalization, Activation, Dense, Input, GlobalMaxPooling3D, GlobalAveragePooling3D, Dropout
from keras.layers import Conv3D, MaxPooling3D
from keras import backend as K
from keras.layers import concatenate, add
from keras.models import Model


# ----------------------------------------------------------------------------------
def _initial_conv_block_inception(input, initial_conv_filters,bias_flag=False):
    ''' Adds an initial conv block, with batch norm and relu for the DPN
    Args:
        input: input tensor
        initial_conv_filters: number of filters for initial conv block
    Returns: a keras tensor
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    #因为体素大小是0.6*0.6*0.8，所以卷积如果想按照真实尺寸正方体卷积，7,7,5是个不错的卷积核size，并且经过这次卷积之后，featuremap的体素就是接近正方体了，便不用再用类似7,7,5这种尺寸了
    x = Conv3D(initial_conv_filters, (7, 7, 5), strides=(2, 2, 1), padding='same', use_bias=bias_flag, kernel_initializer='he_normal', name='init_conv1')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same')(x)

    return x
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
def _bn_relu_conv_block(input, filters, kernel=(3, 3, 3), stride=(1, 1, 1),bias_flag=False):
    ''' Adds a Batchnorm-Relu-Conv block for DPN
    Args:
        input: input tensor
        filters: number of output filters
        kernel: convolution kernel size
        stride: stride of convolution
    Returns: a keras tensor
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv3D(filters, kernel, padding='same', use_bias=bias_flag, kernel_initializer='he_normal', strides=stride)(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# 分组卷积这个函数有一个问题
# input的通道数需要等于grouped_channels*cardinality，也就是说运行函数之前需要计算一下这两个参数，而不是自动计算。
def _grouped_convolution_block_3D(input, grouped_channels, cardinality, strides, bias_flag=False):
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
        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=bias_flag, strides=strides, kernel_initializer='he_normal')(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    # 如果需要分多个组卷积，则标准分组卷积过程如下：
    # 先使用lambda层沿通道轴分割出各个组，之后再卷积，最后再沿通道轴concatenate合并
    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, :, c * grouped_channels:(c + 1) * grouped_channels] if K.image_data_format() == 'channels_last' else lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :, :])(input)
        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=bias_flag, strides=strides, kernel_initializer='he_normal')(x)
        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    group_merge = BatchNormalization(axis=channel_axis)(group_merge)
    group_merge = Activation('relu')(group_merge)
    return group_merge
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
def _dual_path_block(input, pointwise_filters_a, grouped_conv_filters_b, pointwise_filters_c, filter_increment, cardinality, block_type='normal',bias_flag=False):
    '''
    Creates a Dual Path Block. The first path is a ResNeXt type
    grouped convolution block. The second is a DenseNet type dense
    convolution block.
    Args:
        input: input tensor
        pointwise_filters_a: number of filters for the bottleneck pointwise convolution DPN称共同卷积那部分为bottleneck，pointwise_filters_a类似densenet中bottleneck的一个限制作用
        grouped_conv_filters_b: number of filters for the grouped convolution block 需要被分组卷积的输入张量通道数
        pointwise_filters_c: number of filters for the bottleneck convolution block 在共同路径要用到的，类似于指定resnet path的通道数
        filter_increment: number of filters that will be added ，即dense path 每过一个dpn block增加的通道数
        cardinality: cardinality factor 分组卷积的组数，比如分组卷积一共使用90个卷积核，需要分3组卷积，则cardinality=3
        block_type: determines what action the block will perform ，三种类型，具体看网络结构图
            - `projection`: adds a projection connection
            - `downsample`: downsamples the spatial resolution
            - `normal`    : simple adds a dual path connection
    Returns: a list of two output tensors - one path of ResNeXt
        and another path for DenseNet
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    grouped_channels = int(grouped_conv_filters_b / cardinality)  # 计算得出分组卷积每一组里面包括多少卷积核

    # 判断输入是一个list还是别的，如果是list，则代表是[res_path，dense_path]，这样的话需要在通道轴concatenate，
    # 如果不是列表，那就是经过init_block的输入图像，也就是一个tense，这样直接把input赋给init即可
    init = concatenate(input, axis=channel_axis) if isinstance(input, list) else input

    if block_type == 'projection':
        stride = (1, 1, 1)
        projection = True
    elif block_type == 'downsample':  # 注意，整个网络下采样主要就是用这个参数
        stride = (2, 2, 2)
        projection = True
    elif block_type == 'normal':
        stride = (1, 1, 1)
        projection = False
    else:
        raise ValueError('`block_type` must be one of ["projection", "downsample", "normal"]. Given %s' % block_type)

    if projection:
        # 先过一个卷积层
        projection_path = _bn_relu_conv_block(init, filters=pointwise_filters_c + 2 * filter_increment, kernel=(1, 1, 1), stride=stride,bias_flag=bias_flag)

        # 使用 do_1 if A else do_2语法，就是如果满足A则运行do_1，否则运行do_2
        # 下面这两条指令是lambda层的一个用法，实现的功能为分割输入的张量并返回，projection_path即输入张量
        input_residual_path = Lambda(lambda z: z[:, :, :, :, :pointwise_filters_c] if K.image_data_format() == 'channels_last' else z[:, :pointwise_filters_c, :, :, :])(projection_path)
        input_dense_path = Lambda(lambda z: z[:, :, :, :, pointwise_filters_c:] if K.image_data_format() == 'channels_last' else z[:, pointwise_filters_c:, :, :, :])(projection_path)

    else:
        input_residual_path = input[0]
        input_dense_path = input[1]

    x = _bn_relu_conv_block(init, filters=pointwise_filters_a, kernel=(1, 1, 1),bias_flag=bias_flag)
    # 可以看出，grouped_channels*cardinality = pointwise_filters_a，也就是pointwise_filters_a = grouped_conv_filters_b
    x = _grouped_convolution_block_3D(x, grouped_channels=grouped_channels, cardinality=cardinality, strides=stride, bias_flag=bias_flag)
    x = _bn_relu_conv_block(x, filters=pointwise_filters_c + filter_increment, kernel=(1, 1, 1), bias_flag=bias_flag)

    # 下面这两条指令是同样是lambda层的一个用法，实现的功能为分割输入的张量并返回，projection_path即输入张量
    output_residual_path = Lambda(lambda z: z[:, :, :, :, :pointwise_filters_c] if K.image_data_format() == 'channels_last' else z[:, :pointwise_filters_c, :, :, :])(x)
    output_dense_path = Lambda(lambda z: z[:, :, :, :, pointwise_filters_c:] if K.image_data_format() == 'channels_last' else z[:, pointwise_filters_c:, :, :, :])(x)

    residual_path = add([input_residual_path, output_residual_path])
    dense_path = concatenate([input_dense_path, output_dense_path], axis=channel_axis)

    return [residual_path, dense_path]
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

#def dual_path_net(initial_conv_filters = 64, filter_increment = [16, 32, 24, 128], depth = [3, 4, 20, 3], cardinality=32, width=3, pooling='max-avg',bias_flag=True):
def dual_path_net(initial_conv_filters, filter_increment, depth, cardinality, width, pooling='max-avg', bias_flag=False):
    '''
    Args:
        initial_conv_filters: number of features for the initial convolution  初始化输出的张量通道数
        include_top: Flag to include the last dense layer
        initial_conv_filters: number of features for the initial convolution
        filter_increment: number of filters incremented per block, defined as a list.
            DPN-92  = [16, 32, 24, 128]
            DON-98  = [16, 32, 32, 128]
            DPN-131 = [16, 32, 32, 128]
            DPN-107 = [20, 64, 64, 128]
        depth: number or layers in the each block, defined as a list.
            DPN-92  = [3, 4, 20, 3]
            DPN-98  = [3, 6, 20, 3]
            DPN-131 = [4, 8, 28, 3]
            DPN-107 = [4, 8, 20, 3]
        width: width multiplier for network 分组卷积每组的卷积核数量，所以也就是直接指定每组包括多少卷积核和组数即可，不过确实有点绕，因为这样做则要求dpn block函数中的参数pointwise_filters_a和grouped_conv_filters_b相同
        pooling: Optional pooling mode for feature extraction
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
            - `max-avg` means that both global average and global max
                pooling will be applied to the output of the last
                convolution layer
    Returns: a Keras Model
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1  #
    N = list(depth)  #
    base_filters = 256  #

    # input set
    img_input = Input(shape=(280, 280, 16, 1))  #
    # block 1 (initial conv block)
    x = _initial_conv_block_inception(img_input, initial_conv_filters,bias_flag=bias_flag)  #
    print('BLOCK 1 init shape :', x.shape)  #
    # block 2 (projection block)
    filter_inc = filter_increment[0]  #
    # filter_increment: number of filters incremented per block, defined as a list.
    # DPN-92  = [16, 32, 24, 128]
    filters = int(cardinality * width)  #

    x = _dual_path_block(x, pointwise_filters_a=filters, grouped_conv_filters_b=filters, pointwise_filters_c=base_filters, filter_increment=filter_inc, cardinality=cardinality, block_type='projection', bias_flag=bias_flag)  #

    for i in range(N[0] - 1):
        x = _dual_path_block(x, pointwise_filters_a=filters, grouped_conv_filters_b=filters, pointwise_filters_c=base_filters, filter_increment=filter_inc, cardinality=cardinality, block_type='normal', bias_flag=bias_flag)  #

    print("BLOCK 1 out shape : res_path:", x[0].shape, " vs.  dense_path", x[1].shape)  #
    # remaining blocks
    for k in range(1, len(N)):


        filter_inc = filter_increment[k]  #
        filters *= 2  # 进入到下一个大的Block（注意不是dpn block），filters（等于分组卷积的通道数）也要翻倍
        base_filters *= 2  # 这个参数相当于把densepath的通道数改变一下，有点过度模块的意思，因此这个参数要不断增加，因为原始dense net的过度模块也是随着网络深入而逐渐卷积核变多的

        x = _dual_path_block(x, pointwise_filters_a=filters, grouped_conv_filters_b=filters, pointwise_filters_c=base_filters, filter_increment=filter_inc, cardinality=cardinality, block_type='downsample',bias_flag=bias_flag)  #
        print("BLOCK", (k + 1), "d_sample shape : res_path:", x[0].shape, " vs.  dense_path", x[1].shape)  #
        for i in range(N[k] - 1):
            x = _dual_path_block(x, pointwise_filters_a=filters, grouped_conv_filters_b=filters, pointwise_filters_c=base_filters, filter_increment=filter_inc, cardinality=cardinality, block_type='normal',bias_flag=bias_flag)  #

        print("BLOCK", (k + 1), "out shape : res_path:", x[0].shape, " vs.  dense_path", x[1].shape)  #

    x = concatenate(x, axis=channel_axis)  #
    print("CONCAT out shape : ", x.shape)

    if pooling == 'avg':
        x = GlobalAveragePooling3D(data_format='channels_last')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling3D(data_format='channels_last')(x)
    elif pooling == 'max-avg':
        a = GlobalMaxPooling3D(data_format='channels_last')(x)
        b = GlobalAveragePooling3D(data_format='channels_last')(x)
        x = add([a, b])
        x = Lambda(lambda z: 0.5 * z)(x)

    print("GApooling shape:", x.shape)
    out_drop = Dropout(rate=0.3)(x)
    out = Dense(1, name='fc1',use_bias=bias_flag)(out_drop)
    print("out shape:", out.shape)
    output = Activation(activation='sigmoid')(out)

    model = Model(input=img_input, output=output)

    return model


