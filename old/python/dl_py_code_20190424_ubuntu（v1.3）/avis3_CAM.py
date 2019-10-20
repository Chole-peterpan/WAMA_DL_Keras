from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# step1: import model finished
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
H5_file = h5py.File(r'/data/@data_NENs_level_ok/4test/102_1.h5', 'r')
batch_x = H5_file['data'][:]
H5_file.close()
batch_x = np.transpose(batch_x, (1, 2, 0))
data_input_c[0, :, :, :, 0] = batch_x[:, :, :]


# 保存或加載權重
# pre = model.predict_on_batch(data_input_c)
# model.save('G:\qweqweqweqwe\model.h5')
model.load_weights(filepath='/data/XS_Aug_model_result/model_templete/qianyi/model_qianyi_new/2LEVEL/m_80000_model.h5',by_name=True)

# 查看計算圖
print(model.summary())

# 查询自己要做saliency的評分层的index：即求梯度時候的因變量層，一般選最後一個全連接層
layer_idx = utils.find_layer_idx(model, 'fc1')
# 查詢要可視化的層的index：即加權的特征圖所在的層，也是求梯度時候的自變量，一般選最後一個卷積層或池化層
layer_idx_conv = utils.find_layer_idx(model, 'L4_block3conv3')


# 按照套路，首先將評分層激活函數轉換為線性激活
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# 開始CAM
# cam方法的激活 ============================================================================================================
# visualize_cam()用法：
# model：即你想可视化的keras模型
# layer_index：The layer index within `model.layers` whose filters needs to be visualized.
#              官方注释如上，按我的理解，一般这个层应该指定为最后一个全连接层，即最终softmax激活层的前一层，
#              这层的神经元数量等于总类别数，并且各个神经元的输出值相当于对应的每个类别的评分（评分经过softmax激活之后
#              就变成概率值啦）。但是实际可以指定卷积层，这个需要在研究一下代码看看。
# filter_indices：filter indices within the layer to be maximized.
#              官方注释如上，这个值是个索引值（整型），即激活的神经元的索引，也就是对应layer_index这一层的第filter_indices
#              个神经元。指定这个参数，也就是指定最后计算梯度并求各个特征图权重的时候要用到那个神经元的输出作为评分。
#              因为一般指定的“评分层”都是最后一个全连接层，所以这个索引就是代表着第几个类别。但是因为layer_index
#              也可以指定其他层，如非最后一个全连接层或卷积层，此时索引就变成了这个全连接层的第filter_indices个神经元或卷积层
#              的第filter_indices个神经元（不过卷积层这个尚不明确，毕竟一个卷积核包括了许多神经元，需要研究一下代码）
#          ps：官方代码建议，如果可视化最后一个全连接层的CAM的话，最好把softmax激活函数换成线性激活函数。
# seed_input：输入图像，即你想看网络在“某个输入”上的对某一类的CAM，这个就是所谓的“某个输入”
# penultimate_layer_idx：一个索引，代表计算CAM的featumap的输出层（可以是卷积层，池化层，激活层等等），即你想看网络在“某个输入”上对某一类的CAM的时候，
#              每个卷积层关注的区域是不一样的（或者说某一类在不同卷积层的激活区域是不同的），一般来说是越靠近layer_index层的pooling或conv层，关注的区域越接近网络最终关注的区域。
#              当然其他卷积层也可以得到CAM，当你对各个卷积层都做出针对某一类的CAM，一般会发现：随着卷积核所在深度的加深，CAM关注的区域（高热的区域）
#              越来越接近对应类别目标的轮廓（如果你的输入中确实有这个类别的目标或类似这个类别的目标的话，既可以轻松的观察到了）
# backprop_modifier：backprop modifier to use.
#              官方注释如上，即对方向传播修改的方式。
# grad_modifier：gradient modifier to use.
#              官方注释如上，即对梯度的修改方式。貌似是对求出来的梯度进行一些操作，而这些梯度正好用来计算各个featuremap的权重
#
# 上面那俩modifier好像都有guided，relu，None，具体怎么个情况需要研究一下代码和进一步研究一下原理。
# 另外官方给出的函数返回值（也就是CAM）定义很准确：The CAM image indicating the input regions whose change would most contribute towards
#         maximizing the output of `filter_indices`.

# 同樣有三種梯度的處理方式
grads_norm = visualize_cam(model, layer_idx, filter_indices=0, seed_input=data_input_c, backprop_modifier=None, penultimate_layer_idx=layer_idx_conv)
grads_guided = visualize_cam(model, layer_idx, filter_indices=0, seed_input=data_input_c, backprop_modifier='guided', penultimate_layer_idx=layer_idx_conv)
grads_relu = visualize_cam(model, layer_idx, filter_indices=0, seed_input=data_input_c, backprop_modifier='relu', penultimate_layer_idx=layer_idx_conv)


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
