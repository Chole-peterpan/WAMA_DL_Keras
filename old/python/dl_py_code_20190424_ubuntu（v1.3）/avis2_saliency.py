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
from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image    #image：用于图像数据的实时数据增强的（相当）基本的工具集。
import numpy as np
import matplotlib.pyplot as plt
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam, visualize_activation, visualize_cam_with_losses
from keras import activations
import matplotlib.cm as cm
import cv2
import h5py
from __future__ import absolute_import

import numpy as np
from scipy.ndimage.interpolation import zoom

from keras.layers.convolutional import _Conv
from keras.layers.pooling import _Pooling1D, _Pooling2D, _Pooling3D
from keras.layers.wrappers import Wrapper
from keras import backend as K


# from vis import utils
from vis.losses import ActivationMaximization
from vis.backprop_modifiers import get
from vis.optimizer import Optimizer
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
# model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss='binary_crossentropy')


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
# 查询自己要做saliency的层
layer_idx = utils.find_layer_idx(model, 'fc1')


# 开始saliency，具體步驟如下
# 官方的建議是：To visualize activation over final dense layer outputs, we need to switch the softmax
# activation out for linear since gradient of output node will depend on all the other node activations.
# 意思就是說，為了獲得更好的輸出，基於經驗主義，一般會將最後一個全連接層的激活函數變為線性激活，這樣的話
# 我們制定的神經元輸出的值就是對應類的評分，而不是對應類的概率。這個在原理文章里有提到（公式裡面用的就是softmax之前的）
# 另外谷歌那篇做特征最大化的文章也提到了，文章：https://distill.pub/2017/feature-visualization/
# 所以首先我們利用下面這兩行代碼將最後一個全連接層的激活變更為線性（注意，不改變原計算圖，而是copy出一個新的計算圖用來做saliency）
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)
# 三種方式做saliency，根據經驗，最後一個全連接層作為評分層
grads_norm = visualize_saliency(model, layer_idx, filter_indices=1, seed_input=data_input_c, backprop_modifier=None)
grads_guided = visualize_saliency(model, layer_idx, filter_indices=1, seed_input=data_input_c, backprop_modifier='guided')
grads_relu = visualize_saliency(model, layer_idx, filter_indices=1, seed_input=data_input_c, backprop_modifier='relu')


for ii in range(16):
    plt.figure()
    f, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(batch_x[:, :, ii], cmap=plt.cm.gray)
    ax[0, 0].set_title(' or_image', fontsize=14)
    ax[0, 1].imshow(grads_norm[:, :, ii], cmap=plt.cm.jet)
    ax[0, 1].set_title(' grads_norm', fontsize=14)
    ax[1, 0].imshow(grads_guided[:, :, ii], cmap=plt.cm.jet)
    ax[1, 0].set_title(' grads_guided', fontsize=14)
    ax[1, 1].imshow(grads_relu[:, :, ii], cmap=plt.cm.jet)
    ax[1, 1].set_title(' grads_relu', fontsize=14)
    plt.show()









