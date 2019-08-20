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
import prettytable as pt

# from colorama import init, Fore, Back, Style
#
# init(autoreset=False)
# class Colored(object):
#     #  前景色:红色  背景色:默认
#     def red(self, s):
#         return Fore.LIGHTRED_EX + s + Fore.RESET
#     #  前景色:绿色  背景色:默认
#     def green(self, s):
#         return Fore.LIGHTGREEN_EX + s + Fore.RESET
#     def yellow(self, s):
#         return Fore.LIGHTYELLOW_EX + s + Fore.RESET
#     def white(self,s):
#         return Fore.LIGHTWHITE_EX + s + Fore.RESET
#     def blue(self,s):
#         return Fore.LIGHTBLUE_EX + s + Fore.RESET



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


#现在只用这个
#v2.0版本
def test_on_model4_subject(model, test_list, iters,
                           data_input_shape, label_shape,
                           id_savepath, label_savepath, pre_savepath,
                           label_index = 'label_2'):
    """
    
    :param model: 
    :param test_list: 
    :param iters: 
    :param save_path: 
    :param data_input_shape: 
    :param label_shape: 
    :param front_name: 
    :param id_savepath: 
    :param label_savepath: 
    :param pre_savepath: 
    :param file_sep: 
    :param label_index: 
    :return:
     
    保存预测值,同时保存最终指标
    精确到样本,指标计算单位为样本
    ps*写函数注释的时候,三个爽引号后直接回车就行,就会出现以上pycharm自动补充的函数说明
    """

    # 初始化列表,这些列表就是最后要添加到txt文件中的,分别是病人的id,label,预测值
    id_list=[]
    lb_list=[]
    pr_list=[]

    # 构建容器
    data_input_1 = np.zeros([1] + data_input_shape + [1], dtype=np.float32)  # net input container
    label_input_1 = np.zeros([1] + label_shape)


    testtset_num = len(test_list)
    Num_list_test = list(range(testtset_num))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    true_label = []
    pred_value = []


    loss_temp = []

    # 先储存所有结果(之后的代码会将这些patch的结果整合到以并认为单位的结果,不要着急)
    for read_num in Num_list_test:
        read_name = test_list[read_num]
        print(read_name)
        H5_file = h5py.File(read_name, 'r')
        batch_x = H5_file['data'][:]
        batch_y = H5_file[label_index][:]
        print(batch_y)
        batch_x_t = np.transpose(batch_x, (1, 2, 0))
        data_input_1[0, :, :, :, 0] = batch_x_t[:, :, 0:16]
        label_input_1[0] = batch_y
        H5_file.close()

        result_pre = model.predict_on_batch(data_input_1)
        result_loss = model.test_on_batch(data_input_1, label_input_1)

        loss_temp.append(float(result_loss[0]))
        pred_value.append(float(result_pre[0][0]))#可能需要改,根据labelsize来变动,之后做多分类什么的也需要改
        true_label.append(float(batch_y[0][0]))#可能需要改,根据labelsize来变动,之后做多分类什么的也需要改


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
	
    # 整合patch到一病人为单位:
    # 根据样本序号分配并重新加入最终list，最后根据这个最终list来计算最终指标
    final_true_label = []
    final_pred_value = []
    patient_index.sort(reverse=False)
    for patient_id in patient_index: # 首先遍历所有病人序号
        tmp_patient_prevalue = []
        tmp_patient_reallabel = []
        for read_num in Num_list_test:
            # 在每个病人序号下,遍历所有patch的文件名对应的病例号,
            # 如果属于该病人,则append到临时的list,最终在对这个临时的list操作,得到属于这个病人的一个值
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
		
        # txt_id.write(str(float(p_value))+'\n')
        # txt_lb.write(str(float(t_label)) + '\n')
        # txt_pr.write(str(float(ptnt_id)) + '\n')
		
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






    # =====================================================
    mean_loss = np.mean(loss_temp)


    print('Sensitivity', Sensitivity)
    print('Specificity', Specificity)
    print('Accuracy', Accuracy)
    print('Loss', mean_loss)

    # txt_s4.write('acc:'+str(Accuracy) + '\n')
    # txt_s4.write('spc:' + str(Specificity) + '\n')
    # txt_s4.write('sen:' + str(Sensitivity) + '\n')
    # txt_s4.write('auc:' + str(Aucc) + '\n')
    # txt_s4.write('loss:' + str(mean_loss) + '\n')


    txt_id = open(id_savepath, 'a')
    txt_lb = open(label_savepath, 'a')
    txt_pr = open(pre_savepath, 'a')

    txt_id.write(str(iters) + '@' + str(patient_index) + '\n')
    txt_lb.write(str(iters) + '@' + str(final_true_label) + '\n')
    txt_pr.write(str(iters) + '@' + str(final_pred_value) + '\n')

    txt_id.close()
    txt_lb.close()
    txt_pr.close()


    return [Accuracy, Sensitivity, Specificity, Aucc, mean_loss]






















def test_on_model4_subject4_or_train(model, test_list, data_input_shape, label_shape,label_index='label_2'):
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






# 从这个函数开始,就要规范一下写代码的格式了
# 必要的参数之类的解释,必须要加上
def get_param_from_txt(txt_path, print_param = True):
    """
    _
    Args:
        txt_path:the txt file path which contains all params for run,
            the txt writting format follow as : Param_name@Param_value
            the Param_names are:
                 
            - 01) init_lr: 初始学习率
            - 02) optm_sw_iter: 优化器转换的迭代数
            - 03) model_sv_step: 每迭代model_sv_step,保存一次模型以作为newest的模型,
                                这么做的目的是为了在实验意外中断后可load迭代数最近一次的权重以继续实验
                                ps,此参数对应的保存方式类似于save,而不是save as
            - 04) model_sv_step_4_vis: 每迭代model_sv_step_4_vis,保存一次模型以用来可视化
                                ps,此参数对应的保存方式类似于save as,而不是save
            - 05) os_stage:跑代码的平台,windows则为 'W',linux则为 'L'
            - 06) batch_size
            - 07) max_iter: 训练的最大迭代数,大于次数则停止训练
            - 08) min_verify_iter: 开始验证和测试的最小迭代数量,因为main的代码是训练的同时进行测试的
            - 09) ver_step: 当迭代次数大于min_verify_iter时候,每迭代ver_step次,进行一次测试和验证
            - 10) data_input_shape: 输入样本的shape
            - 11) label_index: h5文件的索引,代表样本的label
            - 12) label_shape: label的shape,也相当于label的元素数量,如果label是独热编码3分类标签,则此处为3
                 
            - 13) aug_subject_path:扩增样本的所在路径(如果使用实时扩增,则次参数必须指定为 None)
            - 14) or_subject_path: 未扩增的原始样本所在的文件夹
            - 15) folder_path: 交叉验证分折文件存放的文件夹(如果不是交叉验证实验,则无需指定)
            - 16) result_save_path: log以及model保存的路径
            - 17) fold_name :指定实验对应第几折(如果不是交叉验证实验,则无需制定)
            - 18) task_name: 指定实验名称
            - 19) GPU_index: 指定GPU的序号
       
    Return:字典形式返回各个参数
        
        
        
    txt Example:
# 纯数字参数
init_lr@1e-6
optm_sw_iter@2000
model_sv_step@500                 
model_sv_step_4_vis@5000
batch_size@6
max_iter@20000
min_verify_iter@50
ver_step@50

# list类型参数
data_input_shape@[280,280,16]
label_shape@[2]


# 字符串类型参数
label_index@label3
aug_subject_path@/data/@data_pnens_zhuanyi_dl/data_aug/v
or_subject_path@/data/@data_pnens_zhuanyi_dl/data/v
folder_path@/data/@data_pnens_zhuanyi_dl/CVfold5_newnew
result_save_path@/data/XS_Aug_model_result/model_templete/recurrent/pnens_zhuanyi_resnet_v_new(fuxk)/fold6
fold_name@1
task_name@recurrent fold5
os_stage@L
   
    """

    # 首先读取txt文件到一个list里面(非txt的无后缀文件也可)
    str_list = []
    f = open(txt_path, "r")
    contents = f.readlines()
    for item in contents:
        content = item.strip()
        str_list.append(content)
    f.close()

    # 将list中的字符串提取为参数,并储存到字典中
    # 注意,因为文件路径或者任务名称中可能会有@,所以要从第一个@处开始分割
    dic = {}
    tb = pt.PrettyTable()
    tb.field_names = ["param_name", "param_value","value_type"]
    for tmp_str in str_list:
        param_name = tmp_str.split('@')[0]  # 第一种截取字符串的方法
        param_value = tmp_str[tmp_str.find('@') + 1:]  # 第二种截取字符串的方法

        # 根据参数名,判断是否需要转换为数字,并存储
        if (    param_name == 'optm_sw_iter' or
                param_name == 'model_sv_step'or
                param_name == 'model_sv_step_4_vis'or
                param_name == 'min_verify_iter' or
                param_name == 'ver_step'or
                param_name == 'batch_size' or
                param_name == 'max_iter'
            ):
            param_value = int(param_value)
            dic[param_name]=param_value
            tb.add_row([param_name, param_value, typeof(param_value)])
        # 根据参数名,判断是否需要转换为list,并存储
        if (    param_name == 'data_input_shape' or
                param_name == 'label_shape'):
            param_value = str2list_w(param_value)
            dic[param_name]=param_value
            tb.add_row([param_name, param_value, typeof(param_value)])

        # 根据参数名,判断是否可以存储,若是参数则存储
        if (    param_name == 'label_index' or
                param_name == 'aug_subject_path' or
                param_name == 'or_subject_path' or
                param_name == 'folder_path' or
                param_name == 'result_save_path' or
                param_name == 'fold_name' or
                param_name == 'task_name' or
                param_name == 'os_stage' or
                param_name == 'GPU_index'):
            dic[param_name]=param_value
            tb.add_row([param_name, param_value, typeof(param_value)])

        if (    param_name == 'init_lr'):
            param_value = float(param_value)
            dic[param_name]=param_value
            tb.add_row([param_name, param_value, typeof(param_value)])


    if print_param:
        tb.align["param_value"] = "l"
        tb.align["param_name"] = "r"
        # tb.set_style(pt.MSWORD_FRIENDLY)
        print(tb)

    return dic




















#判断变量类型的函数
def typeof(variate):
    type=None
    if isinstance(variate,int):
        type = "int"
    elif isinstance(variate,str):
        type = "str"
    elif isinstance(variate,float):
        type = "float"
    elif isinstance(variate,list):
        type = "list"
    elif isinstance(variate,tuple):
        type = "tuple"
    elif isinstance(variate,dict):
        type = "dict"
    elif isinstance(variate,set):
        type = "set"
    return type










# 字符串转换为list函数,字符串为'[1,2,3]'这种类型
def str2list_w(a):
    a = a[1:]
    a = a[:-1]
    a = a.split(',')[:]
    ll = []
    for cc in a:
        ll.append(int(cc))
    return ll



# 改变字符串颜色的函数
def char_color(s,front,word):
    new_char = "\033[0;"+str(int(word))+";"+str(int(front))+"m"+s+"\033[0m"
    return new_char


