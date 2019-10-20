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




def verify_on_model(model, vre_list ,iters, data_input_shape, front_name):
    # 不保存预测值,只保存最终指标
    # 不精确到样本,指标计算单位为文件
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


def test_on_model(model, test_list,iters, save_path, data_input_shape,front_name):
    # 保存预测值,同时保存最终指标
    # 不精确到样本,指标计算单位为文件
    data_input_1 = np.zeros([1] + data_input_shape + [1], dtype=np.float32)  # net input container

    pred_txt = save_path + '/' +front_name +'predict_' + str(iters) + '.txt'
    orgi_txt = save_path + '/' +front_name +'orginal_' + str(iters) + '.txt'
    file_txt = save_path + '/' +front_name +'filename_' + str(iters) + '.txt'
    value_txt = save_path + '/' + front_name +'result_' + str(iters) + '.txt'

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
        batch_y = H5_file['label_3'][:]
        print(batch_y)
        batch_x_t = np.transpose(batch_x, (1, 2, 0))
        data_input_1[0, :, :, :, 0] = batch_x_t[:, :, 0:16]
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
    if (sum(true_label)==testtset_num) or (sum(true_label)==0):
        Aucc = 0
        print('only one class')

    else:
        Aucc = metrics.roc_auc_score(true_label, pred_value)
        print('AUC', Aucc)

    print('Sensitivity', Sensitivity)
    print('Specificity', Specificity)
    print('Accuracy', Accuracy)

    txt_s4.write('acc:'+str(Accuracy) + '\n')
    txt_s4.write('spc:' + str(Specificity) + '\n')
    txt_s4.write('sen:' + str(Sensitivity) + '\n')
    txt_s4.write('auc:' + str(Aucc) + '\n')

    return [Accuracy, Sensitivity, Specificity, Aucc]


def test_on_model4_subject(model, test_list,iters, save_path, data_input_shape,front_name):
    # 保存预测值,同时保存最终指标
    # 精确到样本,指标计算单位为样本
    data_input_1 = np.zeros([1] + data_input_shape + [1], dtype=np.float32)  # net input container

    pred_txt = save_path + '/' +front_name +'predict_' + str(iters) + '.txt'
    orgi_txt = save_path + '/' +front_name +'orginal_' + str(iters) + '.txt'
    id_txt = save_path + '/' +front_name +'subjectid_' + str(iters) + '.txt'
    value_txt = save_path + '/' + front_name +'result_' + str(iters) + '.txt'

    txt_s1 = open(pred_txt, 'w')
    txt_s2 = open(orgi_txt, 'w')
    txt_s3 = open(id_txt, 'w')
    txt_s4 = open(value_txt, 'w')

    testtset_num = len(test_list)
    Num_list_test = list(range(testtset_num))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    true_label = []
    pred_value = []
    final_true_label = []
    final_pred_value = []

    # 先储存所有结果，之后算出一个分样本的结果
    for read_num in Num_list_test:
        read_name = test_list[read_num]
        H5_file = h5py.File(read_name, 'r')
        batch_x = H5_file['data'][:]
        batch_y = H5_file['label_3'][:]
        print(batch_y)
        batch_x_t = np.transpose(batch_x, (1, 2, 0))
        data_input_1[0, :, :, :, 0] = batch_x_t[:, :, 0:16]
        H5_file.close()

        result_pre = model.predict_on_batch(data_input_1)

        pred_value.append(float(result_pre[:]))
        true_label.append(float(batch_y[:]))

    patient_order = []
    patient_index = []	
    # 算出样本数量和序号
    for read_num in Num_list_test:
        read_name = test_list[read_num]	
        patient_order_temp = read_name.split('/')[-1]#Windows则为\\
        patient_order_temp = patient_order_temp.split('_')[0]
        #patient_order_temp = int(patient_order_temp)
        if patient_order_temp not in  patient_order:
            patient_order.append(patient_order_temp)
            patient_index.append(int(patient_order_temp))
	
    # 根据样本序号分配并重新加入最终list，最后根据这个最终list来计算最终指标	

    patient_index.sort(reverse=False)
    for patient_id in patient_index:
        tmp_patient_prevalue = []
        tmp_patient_reallabel = []
        for read_num in Num_list_test:
            read_name = test_list[read_num]
            tmp_index = read_name.split('/')[-1]
            tmp_index = tmp_index.split('_')[0]
            tmp_index = int(tmp_index)
            if tmp_index == patient_id:
                # tmp_pre_value = pred_value
                tmp_patient_prevalue.append(pred_value[read_num])
                tmp_patient_reallabel.append(true_label[read_num])
        #此时已经获得了对应第patient_id个样本的全部预测值
		#暂时的策略为：计算预测均值，如果均值大于0.5则取最大值，反之取最小值 ； label任取一个加入
        final_true_label.append(tmp_patient_reallabel[0])
        mean_pre = np.mean(tmp_patient_prevalue)
        if 	mean_pre > 0.5:
            final_pred_value.append(np.max(tmp_patient_prevalue))
        elif mean_pre < 0.5:
            final_pred_value.append(np.min(tmp_patient_prevalue))
        elif mean_pre == 0.5:
            final_pred_value.append(0.5)
		    
		
    # 根据最终list来计算最终指标
    patient_num = patient_index.__len__()		
    for nn in range(patient_num):
        t_label = final_true_label[nn]#true label
        p_value = final_pred_value[nn]
        ptnt_id = patient_index[nn]
		
        txt_s1.write(str(float(p_value))+'\n')
        txt_s2.write(str(float(t_label)) + '\n')
        txt_s3.write(str(float(ptnt_id)) + '\n')
		
        p_label = sigmoid_y(p_value)
		
        if (t_label == 1) and (t_label == p_label):
            tp = tp + 1  # 真阳
        elif (t_label == 0) and (t_label == p_label):
            tn = tn + 1  # 真阴
        elif (t_label == 1) and (p_label == 0):
            fn = fn + 1  # 假阴
        elif (t_label == 0) and (p_label == 1):
            fp = fp + 1  # 假阳    
    
    Sensitivity = tp / ((tp + fn)+(1e-16))
    Specificity = tn / ((tn + fp)+(1e-16))
    Accuracy = (tp + tn) / ((tp + tn + fp + fn)+(1e-16))

    if (sum(final_true_label)==patient_num) or (sum(final_true_label)==0):
        Aucc = 0
        print('only one class')

    else:
        Aucc = metrics.roc_auc_score(final_true_label, final_pred_value)
        print('AUC', Aucc)



    print('Sensitivity', Sensitivity)
    print('Specificity', Specificity)
    print('Accuracy', Accuracy)

    txt_s4.write('acc:'+str(Accuracy) + '\n')
    txt_s4.write('spc:' + str(Specificity) + '\n')
    txt_s4.write('sen:' + str(Sensitivity) + '\n')
    txt_s4.write('auc:' + str(Aucc) + '\n')

    return [Accuracy, Sensitivity, Specificity, Aucc]



# 输入参数为:
# data_path:另一mode数据所在文件夹
# data_fullname:以及当前mode的图像文件名(完整路径)
# file_num_step为命名时候的序号差,比如样本1动脉其1静脉其301,data_fullname是1,则file_num_step为300
def name2othermode(data_path, data_fullname, file_num_step):
    tmp_str = data_fullname.split('/')[-1]
    subject_id_tmp = tmp_str.split('.')[0]
    subject_id = subject_id_tmp.split('_')[0]
    leave_str = tmp_str[len(subject_id):]
    final_path = data_path + '/' + str(int(subject_id)+file_num_step) + leave_str
    return  final_path



# othermode_path:静脉其数据test的文件夹
def test_on_model4_subject_4multi(model, test_list, iters, save_path, data_input_shape, front_name, othermode_path):
    # 保存预测值,同时保存最终指标
    # 精确到样本,指标计算单位为样本
    data_input_1 = np.zeros([1] + data_input_shape + [1], dtype=np.float32)  # net input container
    data_input_2 = np.zeros([1] + data_input_shape + [1], dtype=np.float32)  # net input container

    pred_txt = save_path + '/' + front_name + 'predict_' + str(iters) + '.txt'
    orgi_txt = save_path + '/' + front_name + 'orginal_' + str(iters) + '.txt'
    id_txt = save_path + '/' + front_name + 'subjectid_' + str(iters) + '.txt'
    value_txt = save_path + '/' + front_name + 'result_' + str(iters) + '.txt'

    txt_s1 = open(pred_txt, 'w')
    txt_s2 = open(orgi_txt, 'w')
    txt_s3 = open(id_txt, 'w')
    txt_s4 = open(value_txt, 'w')

    testtset_num = len(test_list)
    Num_list_test = list(range(testtset_num))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    true_label = []
    pred_value = []
    final_true_label = []
    final_pred_value = []

    # 先储存所有结果，之后算出一个分样本的结果
    for read_num in Num_list_test:
        # a data read -----------------
        read_name = test_list[read_num]
        print(read_name)
        H5_file = h5py.File(read_name, 'r')
        batch_x = H5_file['data'][:]
        batch_y = H5_file['label_3'][:]
        print(batch_y)
        batch_x_t = np.transpose(batch_x, (1, 2, 0))
        data_input_1[0, :, :, :, 0] = batch_x_t[:, :, 0:16]
        H5_file.close()

        # a data read -----------------
        read_name_v = name2othermode(othermode_path, read_name, 300)
        print(read_name_v)
        H5_file = h5py.File(read_name_v, 'r')
        batch_x = H5_file['data'][:]
        batch_y1 = H5_file['label_3'][:]
        print(batch_y1)
        batch_x_t = np.transpose(batch_x, (1, 2, 0))
        data_input_2[0, :, :, :, 0] = batch_x_t[:, :, 0:16]
        H5_file.close()




        result_pre = model.predict_on_batch([data_input_1, data_input_2])

        pred_value.append(float(result_pre[:]))
        true_label.append(float(batch_y[:]))

    patient_order = []
    patient_index = []
    # 算出样本数量和序号
    for read_num in Num_list_test:
        read_name = test_list[read_num]
        patient_order_temp = read_name.split('/')[-1]  # Windows则为\\
        patient_order_temp = patient_order_temp.split('_')[0]
        # patient_order_temp = int(patient_order_temp)
        if patient_order_temp not in patient_order:
            patient_order.append(patient_order_temp)
            patient_index.append(int(patient_order_temp))

    # 根据样本序号分配并重新加入最终list，最后根据这个最终list来计算最终指标

    patient_index.sort(reverse=False)
    for patient_id in patient_index:
        tmp_patient_prevalue = []
        tmp_patient_reallabel = []
        for read_num in Num_list_test:
            read_name = test_list[read_num]
            tmp_index = read_name.split('/')[-1]
            tmp_index = tmp_index.split('_')[0]
            tmp_index = int(tmp_index)
            if tmp_index == patient_id:
                # tmp_pre_value = pred_value
                tmp_patient_prevalue.append(pred_value[read_num])
                tmp_patient_reallabel.append(true_label[read_num])
                # 此时已经获得了对应第patient_id个样本的全部预测值
                # 暂时的策略为：计算预测均值，如果均值大于0.5则取最大值，反之取最小值 ； label任取一个加入
        final_true_label.append(tmp_patient_reallabel[0])
        mean_pre = np.mean(tmp_patient_prevalue)
        if mean_pre > 0.5:
            final_pred_value.append(np.max(tmp_patient_prevalue))
        elif mean_pre < 0.5:
            final_pred_value.append(np.min(tmp_patient_prevalue))
        elif mean_pre == 0.5:
            final_pred_value.append(0.5)

    # 根据最终list来计算最终指标
    patient_num = patient_index.__len__()
    for nn in range(patient_num):
        t_label = final_true_label[nn]  # true label
        p_value = final_pred_value[nn]
        ptnt_id = patient_index[nn]

        txt_s1.write(str(float(p_value)) + '\n')
        txt_s2.write(str(float(t_label)) + '\n')
        txt_s3.write(str(float(ptnt_id)) + '\n')

        p_label = sigmoid_y(p_value)

        if (t_label == 1) and (t_label == p_label):
            tp = tp + 1  # 真阳
        elif (t_label == 0) and (t_label == p_label):
            tn = tn + 1  # 真阴
        elif (t_label == 1) and (p_label == 0):
            fn = fn + 1  # 假阴
        elif (t_label == 0) and (p_label == 1):
            fp = fp + 1  # 假阳

    Sensitivity = tp / ((tp + fn) + (1e-16))
    Specificity = tn / ((tn + fp) + (1e-16))
    Accuracy = (tp + tn) / ((tp + tn + fp + fn) + (1e-16))

    if (sum(final_true_label) == patient_num) or (sum(final_true_label) == 0):
        Aucc = 0
        print('only one class')

    else:
        Aucc = metrics.roc_auc_score(final_true_label, final_pred_value)
        print('AUC', Aucc)

    print('Sensitivity', Sensitivity)
    print('Specificity', Specificity)
    print('Accuracy', Accuracy)

    txt_s4.write('acc:' + str(Accuracy) + '\n')
    txt_s4.write('spc:' + str(Specificity) + '\n')
    txt_s4.write('sen:' + str(Sensitivity) + '\n')
    txt_s4.write('auc:' + str(Aucc) + '\n')

    return [Accuracy, Sensitivity, Specificity, Aucc]





