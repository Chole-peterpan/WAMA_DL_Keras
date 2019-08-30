from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import random
import h5py
from keras.losses import binary_crossentropy,categorical_crossentropy
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# step1: import model finished
from function import pause, test_on_model4_subject, test_on_model4_subject4_or_train, lr_mod, name2othermode
from w_dualpathnet import dual_path_net
from w_resnet import resnet_nobn, se_resnet, resnet_or
from w_dense_net import dense_net,se_dense_net
from w_resnext import  resnext, resnext_or, se_resnext
from w_vggnet import vgg16_w_3d, vgg16_w_3d_gb
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss

import keras.backend as K

# step2: import extra model finished


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))




# 设置文件保存路径
Result_save_Path = r"/data/@data_pnens_recurrent_outside/v"
# 读取文件夹h5文件
H5_path = r"/data/@data_pnens_recurrent_outside/v/4test"
file_name = os.listdir(H5_path)
H5_List = []
for file in file_name:
    if file.endswith('.h5'):
        H5_List.append(os.path.join(H5_path, file))


data_input_shape = [280, 280, 16]
label_index = 'label3'#label3
label_shape = [2]
os_stage = "L"  # W:windows or L:linux
#======================================================================================================================
if os_stage == "W":
    file_sep = r"\\"
elif os_stage == "L":
    file_sep = r'/'
else:
    file_sep = r'/'






#
d_model = resnet_or(use_bias_flag=True,classes=2)
d_model.compile(optimizer=adam(lr=2e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
d_model.load_weights(filepath='/data/XS_Aug_model_result/model_templete/recurrent/pnens_zhuanyi_resnet_v_new(fuxk)/fold5/m_60000_model.h5',
    by_name=True)
test_result = test_on_model4_subject(model=d_model, test_list=H5_List, iters=0, save_path=Result_save_Path,
                                     data_input_shape=data_input_shape, label_shape=label_shape, front_name='test',
                                     file_sep=file_sep, label_index=label_index)


















































