from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# step1: import model finished
from function import pause, test_on_model4_subject, test_on_model4_subject4_or_train
from w_dualpathnet import dual_path_net
from w_resnet import resnet, resnet_nobn, se_resnet
from w_dense_net import dense_net,se_dense_net
from w_resnext import  resnext, resnext_or, se_resnext
from w_vggnet import vgg16_w_3d, vgg16_w_3d_gb
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss
from keras.applications.vgg16 import VGG16
import numpy as np
from vis.visualization import visualize_cam
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Conv3D, MaxPooling3D, ZeroPadding2D, merge,Activation
from keras.optimizers import SGD
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from w_resnet import resnet
from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# step1: import model finished
from function import pause, test_on_model4_subject, test_on_model4_subject4_or_train, lr_mod
from w_dualpathnet import dual_path_net
from w_resnet import resnet, resnet_nobn, se_resnet
from w_dense_net import dense_net,se_dense_net
from w_resnext import  resnext, resnext_or, se_resnext
from w_vggnet import vgg16_w_3d, vgg16_w_3d_gb
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss

import keras.backend as K

# step2: import extra model finished


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))



# 定義網絡
def vgg4_w_3d(classes=2):
    inputs = Input(shape=(280, 280, 16, 1), name='input')
    # Block 1
    x = Conv3D(64, 3, padding='same', name='block1_conv1', kernel_initializer='he_normal')(inputs)
    x = Activation(activation='relu', name='ac1')(x)
    x = Conv3D(64, 3, padding='same', name='block1_conv2', kernel_initializer='he_normal')(x)
    x = Activation(activation='relu', name='ac2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)
    print("block1 shape:", x.shape)

    # Block 2
    x = Conv3D(128, 3, padding='same', name='block2_conv1', kernel_initializer='he_normal')(x)
    x = Activation(activation='relu', name='ac3')(x)
    x = Conv3D(128, 3, padding='same', name='block2_conv2', kernel_initializer='he_normal')(x)
    x = Activation(activation='relu', name='ac4')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(x)
    print("block2 shape:", x.shape)

    # dense
    x = Flatten(name='flatten')(x)
    x = Dense(64, name='fc1', kernel_initializer='he_normal')(x)
    x = Activation(activation='relu', name='ac5')(x)
    print("dense1 shape:", x.shape)
    x = Dense(16, name='fc2', kernel_initializer='he_normal')(x)
    x = Activation(activation='relu', name='ac6')(x)
    print("dense2 shape:", x.shape)
    x = Dense(classes, name='predictions', kernel_initializer='he_normal')(x)
    x = Activation(activation='softmax', name='ac7')(x)
    print("dense3 shape:", x.shape)

    model = Model(inputs=inputs, outputs=x, name='vgg16')

    return model



# 構建網絡
model = resnet(use_bias_flag=True,classes=2)


# 構建輸入
data_input_c = np.zeros([1, 280, 280, 16, 1], dtype=np.float32)
H5_file = h5py.File(r'/data/@data_NENs_level_ok/4test/1_1.h5', 'r')
batch_x = H5_file['data'][:]
H5_file.close()
batch_x = np.transpose(batch_x, (1, 2, 0))
data_input_c[0, :, :, :, 0] = batch_x[:, :, :]


# 保存或加載權重
# pre = model.predict_on_batch(data_input_c)
# model.save('G:\qweqweqweqwe\model.h5')
model.load_weights(filepath='/data/XS_Aug_model_result/model_templete/qianyi/model_qianyi_new/2LEVEL/m_80000_model.h5',by_name=True)


print(model.summary())


# 構建輸出featuremap的結構
feature_output_model = Model(inputs=model.input, outputs=model.get_layer("L1_block1conv2").output)
# 如果有多個輸出,也可如下構建
# feature_output_model = Model(inputs=model.input, outputs=[model.get_layer("block1_conv1").output, model.get_layer("block1_conv2").output])
featuremap = feature_output_model.predict(data_input_c)#使用predict得到该层结果


# 輸出第二個通道的featuremap，并分層顯示
featuremap_slice = featuremap[:, :, :, :, 1]
featuremap_slice = np.squeeze(featuremap_slice, axis=0)
z_axis = featuremap_slice.shape[2]
for ii in range(z_axis):
    featuremap_piece = featuremap_slice[:, :, ii]
    plt.figure()
    plt.imshow(featuremap_piece, cmap=plt.cm.gray)
    plt.show()



