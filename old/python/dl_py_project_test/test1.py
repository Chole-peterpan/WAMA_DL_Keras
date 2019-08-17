from keras.applications.vgg16 import VGG16
import numpy as np
from vis.visualization import visualize_cam
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Conv3D, MaxPooling3D, ZeroPadding2D, merge
from keras.optimizers import SGD
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def vgg4_w_2d(classes=2):
    inputs = Input(shape=(280, 280, 16), name='input')
    # Block 1
    x = Conv2D(64, 3, padding='same', name='block1_conv1', kernel_initializer='he_normal')(inputs)
    x = Activation(activation='relu')(x)
    x = Conv2D(64, 3, padding='same', name='block1_conv2', kernel_initializer='he_normal')(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    print("block1 shape:", x.shape)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1', kernel_initializer = 'he_normal')(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2', kernel_initializer = 'he_normal')(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    print("block2 shape:", x.shape)

    # dense
    x = Flatten(name='flatten')(x)
    x = Dense(64, activation='relu', name='fc1', kernel_initializer = 'he_normal')(x)
    x = Dense(16, activation='relu', name='fc2', kernel_initializer = 'he_normal')(x)
    x = Dense(classes, activation='softmax', name='predictions', kernel_initializer = 'he_normal')(x)

    model = Model(inputs=inputs, outputs=x, name='vgg16')
    model.compile(optimizer=SGD(lr=1e-6), loss='categorical_crossentropy', metrics=['acc'])

    return model

def vgg16_w_3d(classes):
    inputs = Input(shape=(224, 224, 15, 1), name='input')
    # Block 1
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 1), name='block2_pool')(x)

    # Block 3
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 1), name='block3_pool')(x)

    # Block 4
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 1), name='block4_pool')(x)

    # Block 5
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block5_pool')(x)

    # dense
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x, name='vgg16')
    model.compile(optimizer=SGD(lr=1e-6), loss='categorical_crossentropy', metrics=['acc'])

    return model


model = vgg16_w_3d(3)
data_input_c = np.zeros([1,224,224,15,1], dtype=np.float32)  # net input container
grads = visualize_cam(model, 22, filter_indices=0, seed_input=data_input_c[0,:,:,:,:])
heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
plt.imshow(heatmap)


#model = vgg16_w_2d(3)
#data_input_c = np.zeros([1,224,224,1], dtype=np.float32)  # net input container
#grads = visualize_cam(model, 22, filter_indices=0, seed_input=data_input_c[0,:,:,:])

#model = VGG16(weights=None, include_top=True)
#data_input_c = np.zeros([1,224,224,3], dtype=np.float32)  # net input container
#grads = visualize_cam(model, 22, filter_indices=0, seed_input=data_input_c[0,:,:,:])





def asd():
    inputs_1 = Input(shape=(3, 3, 3), name='input1')
    x1 = Conv2D(1, (3, 3), strides=(1, 1), activation='relu', padding='same', name='path1_conv1', use_bias=True)(inputs_1)
    x1 = Conv2D(1, (3, 3), strides=(1, 1), activation='relu', padding='same', name='path1_conv2', use_bias=False)(x1)
    x1 = Conv2D(2, (3, 3), activation='relu', padding='same', name='path1_conv3', use_bias=True)(x1)
    x1 = Conv2D(2, (3, 3), activation='relu', padding='same', name='path1_conv4', use_bias=False)(x1)

    inputs_2 = Input(shape=(3, 3, 3), name='input2')
    x2 = Conv2D(1, (3, 3), strides=(1, 1), activation='relu', padding='same', name='path2_conv1', use_bias=True)(inputs_2)
    x2 = Conv2D(1, (3, 3), strides=(1, 1), activation='relu', padding='same', name='path2_conv2', use_bias=False)(x2)
    x2 = Conv2D(2, (3, 3), activation='relu', padding='same', name='path2_conv3', use_bias=True)(x2)
    x2 = Conv2D(2, (3, 3), activation='relu', padding='same', name='path2_conv4', use_bias=False)(x2)

    out = merge([x1, x2], mode='sum')

    model = Model(inputs=[inputs_1, inputs_2], outputs=out, name='dual input net')
    return model

qwe = asd()
print(qwe.summary())
data_input_c = np.zeros([1,3,3,3], dtype=np.float32)+32  # net input container
predictions = qwe.predict([data_input_c, data_input_c])




from keras.applications.vgg16 import VGG16
import numpy as np
from vis.visualization import visualize_cam
from keras.models import *
from keras.layers import Input, Conv2D, Dense
from keras.layers.core import Flatten
from keras.initializers import Constant, RandomNormal, RandomUniform,\
     TruncatedNormal, VarianceScaling, Orthogonal, Identity, lecun_uniform,\
     glorot_normal,glorot_uniform,he_normal,lecun_normal,he_uniform
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def testnet():
    inputs_1 = Input(shape=(16, 16, 1), name='input1')
    x1 = Conv2D(100, (3, 3), name='conv1', use_bias=False, kernel_initializer='Zeros')(inputs_1)
    x1 = Conv2D(100, (3, 3), name='conv2', use_bias=False, kernel_initializer='Zeros')(x1)
    x1 = Conv2D(100, (3, 3), name='conv3', use_bias=False, kernel_initializer='Ones')(x1)
    x1 = Conv2D(100, (3, 3), name='conv4', use_bias=False, kernel_initializer=Constant(value=1))(x1)
    x1 = Conv2D(100, (3, 3), name='conv5', use_bias=False, kernel_initializer=RandomNormal)(x1)
    x1 = Conv2D(100, (3, 3), name='conv6', use_bias=False, kernel_initializer=RandomUniform)(x1)
    x1 = Conv2D(100, (3, 3), name='conv7', use_bias=False, kernel_initializer=TruncatedNormal)(x1)
    x1 = Conv2D(100, (3, 3), name='conv8', use_bias=False, kernel_initializer=VarianceScaling)(x1)
    x1 = Conv2D(100, (3, 3), name='conv9', use_bias=False, kernel_initializer=Orthogonal)(x1)
    x1 = Conv2D(100, (3, 3), name='conv10', use_bias=False, kernel_initializer=Identity)(x1)
    x1 = Conv2D(100, (3, 3), name='conv11', use_bias=False, kernel_initializer=lecun_uniform)(x1)
    x1 = Conv2D(100, (3, 3), name='conv12', use_bias=False, kernel_initializer=glorot_normal)(x1)
    x1 = Conv2D(100, (3, 3), name='conv13', use_bias=False, kernel_initializer=glorot_uniform)(x1)
    x1 = Conv2D(100, (3, 3), name='conv14', use_bias=False, kernel_initializer=he_normal)(x1)
    x1 = Conv2D(100, (3, 3), name='conv15', use_bias=False, kernel_initializer=lecun_normal)(x1)
    x1 = Conv2D(100, (3, 3), name='conv16', use_bias=False, kernel_initializer=he_uniform)(x1)
    x1 = Flatten()(x1)
    out = Dense(2, name='dense1',use_bias=False, kernel_initializer='he_normal')(x1)
    model = Model(inputs=inputs_1, outputs=out, name='dual input net')
    return model

model = testnet()
print(model.summary())


conv1 = model.get_layer(name='conv2')
conv1_weight = conv1.get_weights()[0]
conv1_weight = np.reshape(conv1_weight, conv1_weight.size)
conv1_bias = conv1.get_weights()[1]
plt.subplot(1, 2, 1)
plt.hist(conv1_weight, bins=200, range=[2*conv1_weight.min(), 2*conv1_weight.max()])
plt.title('Zeros')
plt.show()


















