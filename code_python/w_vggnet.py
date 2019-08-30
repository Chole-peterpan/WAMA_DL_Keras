from keras.models import *
from keras.layers import Input, Flatten, Dense, Conv3D, MaxPooling3D, GlobalAveragePooling3D, Dropout, Activation,BatchNormalization
from keras.regularizers import l2

def vgg16_w_3d(classes=2,dropout_rate=0.3,use_bias_flag=False):
    inputs = Input(shape=(280, 280, 16, 1), name='input')
    # Block 1
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1', use_bias=use_bias_flag)(inputs)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv2', use_bias=use_bias_flag)(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)
    print("block1 shape:", x.shape)

    # Block 2
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv1', use_bias=use_bias_flag)(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv2', use_bias=use_bias_flag)(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 1), name='block2_pool')(x)
    print("block2 shape:", x.shape)
    # Block 3
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv1', use_bias=use_bias_flag)(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv2', use_bias=use_bias_flag)(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv3', use_bias=use_bias_flag)(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 1), name='block3_pool')(x)
    print("block3 shape:", x.shape)
    # Block 4
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv1', use_bias=use_bias_flag)(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv2', use_bias=use_bias_flag)(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv3', use_bias=use_bias_flag)(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(x)
    # x = BatchNormalization(axis=-1, epsilon=1e-6, )(x)
    print("block4 shape:", x.shape)
    # Block 5
    # x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv1', use_bias=use_bias_flag)(x)
    # x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv2', use_bias=use_bias_flag)(x)
    # x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv3', use_bias=use_bias_flag)(x)
    # x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block5_pool')(x)
    # print("block5 shape:", x.shape)
    # dense
    x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1', use_bias=use_bias_flag)(x)
    # print("Dense1 shape:", x.shape)
    x = Dropout(rate=dropout_rate)(x)
    x = Dense(128, activation='relu', name='fc2', use_bias=use_bias_flag)(x)
    print("Dense2 shape:", x.shape)
    # x = Dropout(rate=dropout_rate)(x)

    if classes == 1:
        output = Dense(classes, activation='sigmoid', use_bias=use_bias_flag, name='predictions')(x)
        print("predictions1 shape:", output.shape, 'activation:sigmoid')
    else:
        output = Dense(classes, activation='softmax', use_bias=use_bias_flag, name='predictions')(x)
        print("predictions2 shape:", output.shape, 'activation:softmax')


    model = Model(inputs=inputs, outputs=output, name='vgg16')

    return model


def vgg16_w_3d_gb(classes=2, dropout_rate=0.3, use_bias_flag=False, weight_decay=1e-4):
    inputs = Input(shape=(280, 280, 16, 1), name='input')
    # Block 1
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(inputs)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv2', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)
    print("block1 shape:", x.shape)


    # Block 2
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv1', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv2', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 1), name='block2_pool')(x)
    print("block2 shape:", x.shape)
    # Block 3
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv1', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv2', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv3', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 1), name='block3_pool')(x)
    print("block3 shape:", x.shape)
    # Block 4
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv1', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv2', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv3', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(x)
    print("block4 shape:", x.shape)
    # Block 5
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv1', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv2', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv3', use_bias=use_bias_flag, kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block5_pool')(x)
    print("block5 shape:", x.shape)


    x = GlobalAveragePooling3D(data_format='channels_last')(x)
    print("Gpooling shape:", x.shape)
    x = Dropout(rate=dropout_rate)(x)

    if classes == 1:
        output = Dense(classes, activation='sigmoid', use_bias=use_bias_flag, name='predictions')(x)
        print("predictions1 shape:", output.shape, 'activition:sigmoid')
    else:
        output = Dense(classes, activation='softmax', use_bias=use_bias_flag, name='predictions')(x)
        print("predictions2 shape:", output.shape, 'activition:softmax')


    model = Model(inputs=inputs, outputs=output, name='vgg16')

    return model






