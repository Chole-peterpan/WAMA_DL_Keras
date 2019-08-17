from keras.models import *
from keras.layers import Input, Flatten, Dense, Conv3D, MaxPooling3D, GlobalAveragePooling3D, Dropout, Activation


def vgg16_w_3d(classes=2):
    inputs = Input(shape=(280, 280, 16, 1), name='input')
    # Block 1
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)
    print("block1 shape:", x.shape)

    # Block 2
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 1), name='block2_pool')(x)
    print("block2 shape:", x.shape)
    # Block 3
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 1), name='block3_pool')(x)
    print("block3 shape:", x.shape)
    # Block 4
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(x)
    print("block4 shape:", x.shape)
    # Block 5
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block5_pool')(x)
    print("block5 shape:", x.shape)
    # dense
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    print("Dense1 shape:", x.shape)
    x = Dense(2021, activation='relu', name='fc2')(x)
    print("Dense2 shape:", x.shape)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    print("out shape:", x.shape)
    model = Model(inputs=inputs, outputs=x, name='vgg16')

    return model

def vgg16_w_3d_gb(classes=2):
    inputs = Input(shape=(280, 280, 16, 1), name='input')
    # Block 1
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)
    print("block1 shape:", x.shape)


    # Block 2
    #x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    #x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    #x = MaxPooling3D((2, 2, 2), strides=(2, 2, 1), name='block2_pool')(x)
    #print("block2 shape:", x.shape)
    # Block 3
    #x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    #x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    #x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    #x = MaxPooling3D((2, 2, 2), strides=(2, 2, 1), name='block3_pool')(x)
    #print("block3 shape:", x.shape)
    # Block 4
    #x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    #x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    #x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    #x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(x)
    #print("block4 shape:", x.shape)
    # Block 5
    #x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    #x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    #x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    #x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block5_pool')(x)
    #print("block5 shape:", x.shape)


    out = GlobalAveragePooling3D(data_format='channels_last')(x)
    print("Gpooling shape:", out.shape)
    out_drop = Dropout(rate=0.3)(out)
    out = Dense(classes, name='fc1')(out_drop)
    print("out shape:", out.shape)
    output = Activation(activation='sigmoid')(out)

    model = Model(inputs=inputs, outputs=output, name='vgg16')

    return model






