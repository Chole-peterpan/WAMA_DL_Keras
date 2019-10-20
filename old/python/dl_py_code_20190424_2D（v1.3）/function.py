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
import math


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






#现在只用这个
def test_on_model4_subject(model, test_list, iters, save_path, data_input_shape, label_shape, front_name, file_sep,label_index = 'label_2'):
    # 保存预测值,同时保存最终指标
    # 精确到样本,指标计算单位为样本
    data_input_1 = np.zeros([1] + data_input_shape + [3], dtype=np.float32)  # net input container
    label_input_1 = np.zeros([1] + label_shape)

    pred_txt = save_path + file_sep[0] + front_name + 'predict_' + str(iters) + '.txt'
    orgi_txt = save_path + file_sep[0] + front_name + 'orginal_' + str(iters) + '.txt'
    id_txt = save_path + file_sep[0] + front_name + 'subjectid_' + str(iters) + '.txt'
    value_txt = save_path + file_sep[0] + front_name + 'result_' + str(iters) + '.txt'

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

    loss_temp = []

    # 先储存所有结果，之后算出一个分样本的结果
    for read_num in Num_list_test:
        read_name = test_list[read_num]
        print(read_name)
        H5_file = h5py.File(read_name, 'r')
        batch_x = H5_file['data'][:]
        batch_y = H5_file[label_index][:]
        print(batch_y)

        data_input_1[0, :, :, 0] = batch_x[:, :]
        data_input_1[0, :, :, 1] = batch_x[:, :]
        data_input_1[0, :, :, 2] = batch_x[:, :]


        label_input_1[0] = batch_y
        H5_file.close()

        result_pre = model.predict_on_batch(data_input_1)
        result_loss = model.test_on_batch(data_input_1, label_input_1)



        loss_temp.append(float(result_loss[0]))
        pred_value.append(float(result_pre[0][0]))#可能需要改,根据labelsize来变动,之后做多分类什么的也需要改
        true_label.append(float(batch_y[0][0]))#可能需要改,根据labelsize来变动,之后做多分类什么的也需要改

    mean_loss = np.mean(loss_temp)
    patient_order = []
    patient_index = []	
    # 算出样本数量和序号
    for read_num in Num_list_test:
        read_name = test_list[read_num]	
        patient_order_temp = read_name.split(file_sep[0])[-1]#Windows则为\\
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
            tmp_index = read_name.split(file_sep[0])[-1]
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
    print('Loss', mean_loss)

    txt_s4.write('acc:'+str(Accuracy) + '\n')
    txt_s4.write('spc:' + str(Specificity) + '\n')
    txt_s4.write('sen:' + str(Sensitivity) + '\n')
    txt_s4.write('auc:' + str(Aucc) + '\n')
    txt_s4.write('loss:' + str(mean_loss) + '\n')

    return [Accuracy, Sensitivity, Specificity, Aucc, mean_loss]

def test_on_model4_subject4_or_train(model, test_list, data_input_shape, label_shape,label_index='label_2'):
    # 保存预测值,同时保存最终指标
    # 精确到样本,指标计算单位为样本
    data_input_1 = np.zeros([1] + data_input_shape + [3], dtype=np.float32)  # net input container
    label_input_1 = np.zeros([1] + label_shape)

    testtset_num = len(test_list)
    Num_list_test = list(range(testtset_num))
    loss_temp = []

    # 先储存所有结果，之后算出一个分样本的结果
    for read_num in Num_list_test:
        read_name = test_list[read_num]
        print(read_name)
        H5_file = h5py.File(read_name, 'r')
        batch_x = H5_file['data'][:]
        batch_y = H5_file[label_index][:]
        print(batch_y)

        data_input_1[0, :, :, 0] = batch_x[:, :]
        data_input_1[0, :, :, 1] = batch_x[:, :]
        data_input_1[0, :, :, 2] = batch_x[:, :]
        label_input_1[0] = batch_y
        H5_file.close()

        result_loss = model.test_on_batch(data_input_1, label_input_1)
        loss_temp.append(float(result_loss[0]))

    mean_loss = np.mean(loss_temp)

    return mean_loss









def test_on_model4_subject4grt_weight(model, test_list, iters, save_path, data_input_shape, label_shape, front_name, file_sep,
                           label_index='label_2'):
    # 保存预测值,同时保存最终指标
    # 精确到样本,指标计算单位为样本
    data_input_1 = np.zeros([1] + data_input_shape + [1], dtype=np.float32)  # net input container
    label_input_1 = np.zeros([1] + label_shape)

    pred_txt = save_path + file_sep[0] + front_name + 'predict_' + str(iters) + '.txt'
    orgi_txt = save_path + file_sep[0] + front_name + 'orginal_' + str(iters) + '.txt'
    id_txt = save_path + file_sep[0] + front_name + 'subjectid_' + str(iters) + '.txt'
    value_txt = save_path + file_sep[0] + front_name + 'result_' + str(iters) + '.txt'

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

    loss_temp = []

    # 先储存所有结果，之后算出一个分样本的结果
    for read_num in Num_list_test:
        read_name = test_list[read_num]
        print(read_name)
        H5_file = h5py.File(read_name, 'r')
        batch_x = H5_file['data'][:]
        batch_y = H5_file[label_index][:]
        print(batch_y)
        batch_x_t = np.transpose(batch_x, (1, 2, 0))
        data_input_1[0, :, :, :, 0] = batch_x_t[:, :, 0:16]
        label_input_1[0] = batch_y[0][1:3]
        H5_file.close()

        result_pre = model.predict_on_batch(data_input_1)
        result_loss = model.test_on_batch(data_input_1, label_input_1)

        loss_temp.append(float(result_loss[0]))
        pred_value.append(float(result_pre[0][0]))
        true_label.append(float(batch_y[0][1]))

    mean_loss = np.mean(loss_temp)
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
    print('Loss', mean_loss)

    txt_s4.write('acc:' + str(Accuracy) + '\n')
    txt_s4.write('spc:' + str(Specificity) + '\n')
    txt_s4.write('sen:' + str(Sensitivity) + '\n')
    txt_s4.write('auc:' + str(Aucc) + '\n')
    txt_s4.write('loss:' + str(mean_loss) + '\n')

    return [Accuracy, Sensitivity, Specificity, Aucc, mean_loss]


# 针对or train的测试代码，没有保存，sen，spc之类的东西，只有loss
def test_on_model4_subject4_or_train4_getweight(model, test_list, data_input_shape, label_shape, label_index='label_2'):
    # 保存预测值,同时保存最终指标
    # 精确到样本,指标计算单位为样本
    data_input_1 = np.zeros([1] + data_input_shape + [1], dtype=np.float32)  # net input container
    label_input_1 = np.zeros([1] + label_shape)

    testtset_num = len(test_list)
    Num_list_test = list(range(testtset_num))
    loss_temp = []

    # 先储存所有结果，之后算出一个分样本的结果
    for read_num in Num_list_test:
        read_name = test_list[read_num]
        print(read_name)
        H5_file = h5py.File(read_name, 'r')
        batch_x = H5_file['data'][:]
        batch_y = H5_file[label_index][:]
        print(batch_y)
        batch_x_t = np.transpose(batch_x, (1, 2, 0))
        data_input_1[0, :, :, :, 0] = batch_x_t[:, :, 0:16]
        label_input_1[0] = batch_y[0][1:3]
        H5_file.close()

        result_loss = model.test_on_batch(data_input_1, label_input_1)
        loss_temp.append(float(result_loss[0]))

    mean_loss = np.mean(loss_temp)

    return mean_loss

# 多期像用到的函数
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
        batch_y = H5_file['label_2'][:]
        print(batch_y)
        batch_x_t = np.transpose(batch_x, (1, 2, 0))
        data_input_1[0, :, :, :, 0] = batch_x_t[:, :, 0:16]
        H5_file.close()

        # a data read -----------------
        read_name_v = name2othermode(othermode_path, read_name, 300)
        print(read_name_v)
        H5_file = h5py.File(read_name_v, 'r')
        batch_x = H5_file['data'][:]
        batch_y1 = H5_file['label_2'][:]
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




def lr_mod(iter,max_epoch,epoch_file_size,batch_size,init_lr ,doudong=0.1,min_lr_limitation=2.2,cos_ca=0.3):

    all_batch_num = np.floor(max_epoch * (epoch_file_size / batch_size))
    per_batch_num = np.floor(epoch_file_size / batch_size)
    max_lr = (1+doudong)*init_lr

#如果没有下面这两行代码,曲线会变得很有意思
    if iter>all_batch_num:
        iter = all_batch_num


    pi = math.pi

    value = np.cos(((iter%per_batch_num)/(per_batch_num))*0.5*pi)
    current_epoch = iter/per_batch_num
    init_lr_tmp = init_lr*(1-current_epoch/max_epoch)
    lr= (value+doudong)*init_lr_tmp
    lr = lr+ min_lr_limitation*max_lr
    lr = lr/((1+min_lr_limitation)*max_lr)
    lr= lr*init_lr
    value1 = np.cos((iter/all_batch_num)*cos_ca*pi)
    lr_new = lr*value1

    return lr_new

























