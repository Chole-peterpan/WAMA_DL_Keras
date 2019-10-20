from keras.layers import merge
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D, AveragePooling3D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# step1: import model finished
from function import sigmoid_y, pause, verify_on_model, test_on_model
from net import resnetttt, alexnet_jn  # use as :Resnetttt(tuple(data_input_shape+[1]))
from Net_new import ClassNet
from sklearn import cross_validation,metrics


# batch test
batch_size = 5
max_iter = 100000
H5_train_path1 = r'G:\diploma_project\data_huanhui\@aug_data\G1'  # CD patient is positive
data_input_shape = [280, 280, 16]
label_shape = [1]
filename_train_posi = os.listdir(H5_train_path1)
H5_List_train = []
for file in filename_train_posi:
    if file.endswith('.h5'):
        H5_List_train.append(os.path.join(H5_train_path1, file))

trainset_num = len(H5_List_train)
Num_list_train = list(range(trainset_num))

epoch = 0
index_flag = 0
Iter = 0
for i in range(max_iter):
    Iter = Iter + 1
    print('Iter:', Iter, 'epoch', epoch)
    for ii in range(batch_size):
        # read data from h5
        read_name = H5_List_train[index_flag]
        print(read_name)
        index_flag = index_flag + 1
        if index_flag == trainset_num:
            index_flag = 0
            epoch = epoch +1
    input()


# auc test
test_y = [.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00]
prodict_prob_y = [.60,.37,.81,.41,.55,.33,.20,.31,.42,.36,.33,.19,.57,.74,.41,.68,.60,.35,.68,.52,.57,.63,.73,.78]
test_auc = metrics.roc_auc_score(test_y,prodict_prob_y)#验证集上的auc值
# 转换成array依然可以
test_y_array = np.array(test_y)
prodict_prob_y_array = np.array(prodict_prob_y)
test_auc1 = metrics.roc_auc_score(test_y,prodict_prob_y)#验证集上的auc值


















