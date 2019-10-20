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
from w_resnet import resnet, resnet_nobn, se_resnet,resnet_or
from w_dense_net import dense_net,se_dense_net
from w_resnext import  resnext, resnext_or, se_resnext
from w_vggnet import vgg16_w_3d, vgg16_w_3d_gb
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss
import cv2
import keras.backend as K

# step2: import extra model finished


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))



# 定義網絡

# 構建網絡

# 構建網絡
model = resnet_or(use_bias_flag=True,classes=2)
# 如果是迁移的权重,则需要吧最后一个全里那阶层的名字改成对应的.
# model.layers[-1].name = 'fc_new'

# model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss='binary_crossentropy')


# 構建輸入
data_input_c = np.zeros([1, 280, 280, 16, 1], dtype=np.float32)
H5_file = h5py.File(r'/data/@data_liaoxiao/4test/27_18.h5', 'r')
batch_x = H5_file['data'][:]
batch_y = H5_file['label3'][:]
print(batch_y)
H5_file.close()
batch_x = np.transpose(batch_x, (1, 2, 0))
data_input_c[0, :, :, :, 0] = batch_x[:, :, :]


# 保存或加載權重
# pre = model.predict_on_batch(data_input_c)
# model.save('G:\qweqweqweqwe\model.h5')
model.load_weights(filepath='/data/XS_Aug_model_result/model_templete/diploma_real/@test40/m_40000_model.h5',by_name=True)


# 構建輸入
# data_input_c = np.zeros([1, 280, 280, 16, 1], dtype=np.float32)
# H5_file = h5py.File(r'/data/@data_NENs_level_ok/4test/1_1.h5', 'r')
# batch_x = H5_file['data'][:]
# H5_file.close()
# batch_x = np.transpose(batch_x, (1, 2, 0))
# data_input_c[0, :, :, :, 0] = batch_x[:, :, :]



print(model.summary())


# 構建輸出featuremap的結構
feature_output_model = Model(inputs=model.input, outputs=model.get_layer("L4_block1conv1").output)
# 如果有多個輸出,也可如下構建
# feature_output_model = Model(inputs=model.input, outputs=[model.get_layer("block1_conv1").output, model.get_layer("block1_conv2").output])
featuremap = feature_output_model.predict(data_input_c)#使用predict得到该层结果


# 輸出第二個通道的featuremap，并分層顯示
featuremap_slice = featuremap[:, :, :, :, 26]

featuremap_slice = np.squeeze(featuremap_slice, axis=0)
# cv2.normalize(featuremap_slice, featuremap_slice)


z_axis = featuremap_slice.shape[2]

for ii in range(z_axis):
    featuremap_piece = featuremap_slice[:, :, ii]
    plt.figure()

    plt.imshow(featuremap_piece, cmap=plt.cm.gray)
    plt.show()



