from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import prettytable as pt
from function import *
from w_dualpathnet import dual_path_net
from w_resnet import resnet_nobn, se_resnet, resnet_or
from w_dense_net import dense_net,se_dense_net
from w_resnext import  resnext, resnext_or, se_resnext
from w_vggnet import vgg16_w_3d, vgg16_w_3d_gb
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss

import keras.backend as K


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))



# 读取数据

h5dir = '/data/@data_pnens_recurrent_new/data_aug/a'
file_name = os.listdir(h5dir)
H5_List = []
for file in file_name:
    if file.endswith('.h5'):
        H5_List.append(os.path.join(h5dir, file))




# 构建网络
d_model = resnet_or(use_bias_flag=True,classes=2)
d_model.compile(optimizer=adam(lr=2e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])

data_input_shape = [280,280,16]
data_input_1 = np.zeros([1] + data_input_shape + [1], dtype=np.float32)
# 循环predict康康
for i in range(H5_List.__len__()):
    read_name = H5_List[i]
    print(read_name)
    H5_file = h5py.File(read_name, 'r')
    batch_x = H5_file['data'][:]
    batch_x_t = np.transpose(batch_x, (1, 2, 0))
    data_input_1[0, :, :, :, 0] = batch_x_t[:, :, :]
    H5_file.close()


    d_model.predict(data_input_1)
    print(str(i),'/',str(H5_List.__len__()))

















