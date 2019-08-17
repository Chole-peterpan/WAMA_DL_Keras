from keras.layers import merge
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D, AveragePooling3D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
import os
import numpy as np
import random, h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn import cross_validation,metrics


def sigmoid_y(x):
    if x < 0.5:
        x = 0
    else:
        x = 1
    return x

def pause():
    print('type to continue')
    input()
    return 1

def verify_on_model(model, vre_list ,iters, data_input_shape):
    data_input_1 = np.zeros([1] + data_input_shape + [1], dtype=np.float32)  # net input container
    testtset_num = len(vre_list)
    Num_list_verify = list(range(testtset_num))
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for read_num in Num_list_verify:
        read_name = vre_list[read_num]
        H5_file = h5py.File(read_name, 'r')
        batch_x = H5_file['data'][:]
        batch_y = H5_file['label'][:]
        y = np.reshape(batch_y, [1, 2])
        data_input_1[0, :, :, :, 0] = batch_x[:, :, :]
        H5_file.close()

        result_pre = model.predict_on_batch(data_input_1)
        result_pre[:, 0] = sigmoid_y(result_pre[:, 0])
        result_pre[:, 1] = sigmoid_y(result_pre[:, 1])

        if (y[0, 0] == 0) and (y[0, 0] == result_pre[0, 0]):
            tp = tp + 1
        elif (y[0, 0] == 1) and (y[0, 0] == result_pre[0, 0]):
            tn = tn + 1
        elif (y[0, 0] == 0) and (result_pre[0, 0] == 1):
            fn = fn + 1
        elif (y[0, 0] == 1) and (result_pre[0, 0] == 0):
            fp = fp + 1

        print('Sample_name', read_name)
        # print('Sample_label', y)
        # print('Sample_pre_label', result_pre)
        # print('num', read_num, result)
        # print(d_model.predict_on_batch(data_input_1))
        # accuracy, sensitivity, specificity

    Sensitivity = tp / ((tp + fn)+0.01)
    Specificity = tn / ((tn + fp)+0.01)
    Accuracy = (tp + tn) / ((tp + tn + fp + fn)+0.01)
    print('Sensitivity', Sensitivity)
    print('Specificity', Specificity)
    print('Accuracy', Accuracy)
    print('Iters', iters)

    return [Accuracy, Sensitivity, Specificity]




def test_on_model(model, test_list,iters, save_path, data_input_shape):
    data_input_1 = np.zeros([1] + data_input_shape + [1], dtype=np.float32)  # net input container

    pred_txt = save_path + '/predict_' + str(iters) + '.txt'
    orgi_txt = save_path + '/orginal_' + str(iters) + '.txt'
    file_txt = save_path + '/filename_' + str(iters) + '.txt'
    value_txt = save_path + '/result_' + str(iters) + '.txt'

    txt_s1 = open(pred_txt, 'w')
    txt_s2 = open(orgi_txt, 'w')
    txt_s3 = open(file_txt, 'w')
    txt_s4 = open(value_txt, 'w')

    testtset_num = len(test_list)
    Num_list_test = list(range(testtset_num))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    true_label = []
    pred_value = []

    for read_num in Num_list_test:
        read_name = test_list[read_num]
        H5_file = h5py.File(read_name, 'r')
        batch_x = H5_file['data'][:]
        batch_y = H5_file['label_1'][:]
        batch_x_t = np.transpose(batch_x, (1, 2, 0))
        data_input_1[0, :, :, :, 0] = batch_x_t[:, :, :]
        H5_file.close()

        result_pre = model.predict_on_batch(data_input_1)

        true_label.append(float(batch_y[:]))
        pred_value.append(float(result_pre[:]))
        txt_s1.write(str(float(result_pre))+'\n')
        txt_s2.write(str(float(batch_y)) + '\n')
        txt_s3.write(read_name + '\n')

        y = batch_y
        result_pre = sigmoid_y(result_pre)
        if (y == 1) and (y == result_pre):
            tp = tp + 1  # 真阳
        elif (y == 0) and (y == result_pre):
            tn = tn + 1  # 真阴
        elif (y == 1) and (result_pre == 0):
            fn = fn + 1  # 假阴
        elif (y == 0) and (result_pre == 1):
            fp = fp + 1  # 假阳

        print('Sample_name', read_name)
        # print('Sample_label', y)
        # print('Sample_pre_label', result_pre)
        # print('num', read_num, result)
        # print(d_model.predict_on_batch(data_input_1))
        # accuracy, sensitivity, specificity



    Sensitivity = tp / ((tp + fn)+(1e-16))
    Specificity = tn / ((tn + fp)+(1e-16))
    Accuracy = (tp + tn) / ((tp + tn + fp + fn)+(1e-16))
    Aucc = metrics.roc_auc_score(true_label, pred_value)
    print('Sensitivity', Sensitivity)
    print('Specificity', Specificity)
    print('Accuracy', Accuracy)
    print('AUC', Aucc)
    txt_s4.write('acc:'+str(Accuracy) + '\n')
    txt_s4.write('spc:' + str(Specificity) + '\n')
    txt_s4.write('sen:' + str(Sensitivity) + '\n')
    txt_s4.write('auc:' + str(Aucc) + '\n')

    return [Accuracy, Sensitivity, Specificity, Aucc]

