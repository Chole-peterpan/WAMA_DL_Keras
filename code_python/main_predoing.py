from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from w_dualpathnet import dual_path_net
from w_resnet import resnet_or
from w_dense_net import dense_net,se_dense_net
from w_resnext import  resnext, resnext_or
from w_vggnet import vgg16_w_3d, vgg16_w_3d_gb
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss,EuiLoss_new
import keras.backend as K
from function import *
from sparsenet import SparseNetImageNet121
import random
import math
from keras.losses import categorical_crossentropy
# step2: import extra model finished


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# ======================================================s
# ======================================================
# ======================================================
# ======================================================
# ======================================================
# 读取文件的第一种情况:训练集和测试集分别放在两个文件夹 ======================================================

# train_file_path = r'/data/@data_laowang/@@flow1/3or_h5'
# test_file_path = r'/data/@data_laowang/@@flow1/3or_h5'
#
# # 还可以指定病人id
# train_id =  [26, 43, 15, 47, 6, 14, 7, 10, 9, 4, 11, 37, 30, 22, 1, 33, 40, 3, 34, 17, 18, 2, 49, 5, 19, 23, 42, 8, 29, 20, 41, 46, 55, 50, 59, 52, 56, 51, 54]
# test_id =  [16, 31, 13, 27, 28, 39, 38, 25, 32, 36, 35, 24, 21, 12, 58, 57, 53]
#
#
# H5_List_train = get_filelist_frompath(train_file_path,'h5',train_id)
# H5_List_test = get_filelist_frompath(test_file_path,'h5',test_id)
#
# trainset_num = len(H5_List_train)


# # test_pathouy = r"/data/@data_laowang/@@flow3/3or_h5_outside"
# # test_Listou = get_filelist_frompath(test_pathouy,'h5')
# # H5_List_train = H5_List_train+test_Listou



# 读取文件的第二种情况:把一个文件夹的数据分为训练集和测试集 ===================================================
# 将样本文件路径读到dict里面保存
# file_path = r"/data/@data_laowang/@@flow1/3or_h5"
# H5file_List = get_filelist_frompath(file_path,'h5')
# patient_id = []
# patient_dict = {}
# for file in H5file_List:# 算出样本数量和序号
#     # read_name = file
#     id = file.split('/')[-1]  # Windows则为\\
#     id = int(id.split('_')[0])
#     # patient_order_temp = int(patient_order_temp)
#     if id not in patient_id:
#         patient_id.append(id)
#         patient_dict[id]=[]
#     patient_dict[id].append(file)
# patient_id.sort()
# print(patient_id)
#
#
# class_dict = {}
# # class_dict[0]=list(np.array(list(range(49)))+1)
# # class_dict[1]=list(np.array(list(range(10)))+50)
# class_dict[0]=list(np.array([1,2,3,4,5,6,7,8,9,10,
#                11,12,13,14,15,16,17,18,19,20,
#                21,22,23,24,25,26,27,28,29,30,
#                31,32,33,34,35,36,37,38,39,40,
#                41,42,43,46,47,49]))
# class_dict[1]=list(np.array(list(range(10)))+50)
# # test_rate = 3/10
# # train_id = []
# # test_id = []
# # for class_index in class_dict.keys():
# #     random.shuffle(class_dict[class_index])
# #     class_subject_num = len(class_dict[class_index])
# #     test_id = test_id + class_dict[class_index][0:math.ceil(class_subject_num*test_rate)]
# #     train_id = train_id + class_dict[class_index][math.ceil(class_subject_num*test_rate):]
# # print('train',train_id)
# # print('test',test_id)
#
# train_id =  [26, 43, 15, 47, 6, 14, 7, 10, 9, 4, 11, 37, 30, 22, 1, 33, 40, 3, 34, 17, 18, 2, 49, 5, 19, 23, 42, 8, 29, 20, 41, 46, 55, 50, 59, 52, 56, 51, 54]
# test_id =  [16, 31, 13, 27, 28, 39, 38, 25, 32, 36, 35, 24, 21, 12, 58, 57, 53]
#
#
# H5_List_test = []
# H5_List_train = []
# for id in patient_id:
#     if id not in train_id:
#         H5_List_test = H5_List_test + patient_dict[id]
#     else:
#         H5_List_train = H5_List_train + patient_dict[id]
#
#
# # # 读取外部数据集到list里面
# # test_pathouy = r"/data/@data_laowang/@@flow3/3or_h5_outside"
# # test_Listou = get_filelist_frompath(test_pathouy,'h5')
# # H5_List_train = H5_List_train+test_Listou
#

# trainset_num = len(H5_List_train)




# 读取文件的第三种情况:从分折文件中读取 ===================================================================
file_sep = os.sep
CV_path = '/data/@data_laowang/new/flow1/5CV'
foldname = '3'# 注意修改!!!
task_name = 'fold3'  # 任务名称,自己随便定义# 注意修改!!!
augdata_path = '/data/@data_laowang/new/flow1/4aug'
or_data_path = '/data/@data_laowang/new/flow1/3or'
out_or_path  = '/data/@data_laowang/new/flow1/3or_out'
Result_save_Path = r'/data/@data_laowang/new/result/test'# 注意修改!!!



train_list1 =get_filelist_fromTXT(augdata_path,CV_path+file_sep+'fold_'+foldname+'_aug_train.txt')
train_list2 =get_filelist_fromTXT(augdata_path,CV_path+file_sep+'fold_'+foldname+'_aug_verify.txt')
train_list3 =get_filelist_fromTXT(or_data_path,CV_path+file_sep+'fold_'+foldname+'_or_train.txt')
train_list4 =get_filelist_fromTXT(or_data_path,CV_path+file_sep+'fold_'+foldname+'_or_verify.txt')

test_list =get_filelist_fromTXT(or_data_path,CV_path+file_sep+'fold_'+foldname+'_or_test.txt')
ver_list = get_filelist_frompath(out_or_path,'h5')



H5_List_train =  train_list3 + train_list4 +train_list1+train_list2
H5_List_test = test_list
H5_List_ver = ver_list

trainset_num = len(H5_List_train)









# ======================================================
# ======================================================
# ======================================================
# ======================================================
# ======================================================


multi_gpu_mode = False #是否开启多GPU数据并行模式
gpu_munum = 3 #多GPU并行指定GPU数量


batch_size = 7
test_batchsize = 4
label_index = 'label3'
data_input_shape = [200,200,20]
label_shape = [2]
init_lr = 2e-4 # 初始学习率
trans_lr = 2e-5 # 转换后学习率
max_iter = 400000 #最大迭代次数
print_steps = 50 # 打印训练信息的步长

min_verify_Iters = 500  # 最小测试和验证迭代数量
verify_Iters_step = 100 # 测试和验证的步长
# lr_decay_step = 10000
# decay_rate = 0.5
model_save_step = 500 # 模型保存的步长
model_save_step4_visualization = 2000 # 保存模型用来可视化的步长

optimizer_switch_point = 2000000 # 优化器转换的迭代数时间点


log_save_Path = Result_save_Path + file_sep + 'log'
test_log_save_path = log_save_Path + file_sep + '@test'
ver_log_save_path = log_save_Path + file_sep + '@ver'
model_save_Path = Result_save_Path + file_sep + '@model'

os.system('mkdir '+log_save_Path)
os.system('mkdir '+model_save_Path)
os.system('mkdir '+test_log_save_path)
os.system('mkdir '+ver_log_save_path)
# 开始训练:一下代码由main改变而来,但是不需要保存模型


if __name__ == "__main__":
    # prepare container:准备网络输入输出容器 暂时测试新getbatch函数,所以注释掉了
    # data_input_c = np.zeros([batch_size] + data_input_shape + [1], dtype=np.float32)  # net input container
    # label_input_c = np.zeros([batch_size] + label_shape)


    # 构建网络并compile
    # d_model_1 = vgg16_w_3d(use_bias_flag=True,classes=2)
    d_model = resnet_or(use_bias_flag=False,classes=2,inputshape=data_input_shape)
    # d_model_1 = resnext(classes=2, use_bias_flag=True)
    # d_model_1 = SparseNetImageNet121(classes = 2,activation='softmax',dropout_rate = 0.5)

    # d_model_1 = se_dense_net(nb_layers=[6, 12, 24, 16], growth_rate=32, nb_filter=64, bottleneck=True, reduction=0.0, dropout_rate=None, subsample_initial_block=True)


    if multi_gpu_mode:
        d_model = multi_gpu_model(d_model, gpus=gpu_munum)
    d_model.compile(optimizer=adam(lr=init_lr), loss=categorical_crossentropy, metrics=[y_t, y_pre, Acc])
    # d_model.compile(optimizer=SGD(lr=init_lr, momentum=0.9), loss=categorical_crossentropy, metrics=[y_t, y_pre, Acc])
    # d_model.compile(optimizer=SGD(lr=init_lr,momentum=0.9), loss='categorical_crossentropy', metrics=[y_t, y_pre, Acc])
    # pause()  # identify
    # print(d_model.summary())  # view net
    # pause()  # identify
    # extra param initialization:初始化一些用来记录和显示的参数


    # 寻找最佳学习率
    random.shuffle(H5_List_train)
    finder_loss, finder_loss_smooth, finder_lr = lr_finder(d_model,
                                                           Result_save_Path,
                                                           H5_List_train,
                                                           batch_size, 2, label_shape,
                                                           data_input_shape,
                                                           label_index,
                                                           inc_mode='mult',
                                                           show_flag=True,
                                                           iter = 300,
                                                           lr_low=1e-12,
                                                           lr_high=1)


    # 键盘手动输入lr_high和low
    keyboard_input=input(r'pls enter lr_high and lr_low') # exp.   1e-6, 1e-11
    lr_high,lr_low = [float(i)  for i in (keyboard_input.split(',')[:])]
    # 获取cos退火的学习率list,注意,这里的epoch file size以未扩增的为准
    lr_sgdr = lr_mod_cos(epoch_file_size = len(train_list3)+len(train_list4),
                         batchsize = batch_size,
                         lr_high = lr_high,
                         lr_low = lr_low,
                         warmup_epoch=5,
                         loop_step=[1, 2, 4, 16],
                         max_contrl_epoch=165,
                         show_flag= True)

    # 初始化一些参数
    have_test_flag = False # 是否已经进行过测试的flag,如果已经测试过,那么就么必要在保存name和id之类的了
    Iter = 0
    epoch = 0
    max_acc_verify = 0.61
    max_acc_verify_iter = 61
    max_acc_test = 0.62
    max_acc_test_iter = 62

    max_auc_verify = 0.63
    max_auc_verify_iter = 63
    max_auc_test = 0.64
    max_auc_test_iter = 64

    max_spc_verify = 0.65
    max_spc_verify_iter = 65
    max_spc_test = 0.66
    max_spc_test_iter = 66

    max_sen_verify = 0.67
    max_sen_verify_iter = 67
    max_sen_test = 0.68
    max_sen_test_iter = 68

    min_loss_verify = 51
    min_loss_verify_iter = 51
    min_loss_test = 52
    min_loss_test_iter = 52


    vre_result = [0, 0, 0, 0]  # acc,sen,spc,auc
    test_result = [0, 0, 0, 0]  # acc,sen,spc,auc
    subject_num = 0
    # txt as a log
    lr_txt = log_save_Path + file_sep + '@' + foldname + '_lr.txt'  # 学习率曲线  txt_s11
    minibatch_loss_txt = log_save_Path + file_sep+'@' + foldname + '_loss.txt'  # minibatch上的loss  txt_minibatch_loss
    or_loss_txt = log_save_Path + file_sep+'@' + foldname + '_loss_or.txt'  # 在原未扩增训练集上的loss  txt_or_loss
    ver_loss_txt = log_save_Path + file_sep+'@' + foldname + '_loss_ver.txt'  # 在验证集上的loss  txt_ver_loss
    test_loss_txt = log_save_Path + file_sep+'@' + foldname + '_loss_test.txt'  # 在测试集上的loss  txt_test_loss

    verify_result_txt = log_save_Path + file_sep+'@' + foldname + '_result_ver.txt'  # 验证集的所有指标  txt_ver_result
    test_result_txt = log_save_Path + file_sep+'@' + foldname + '_result_test.txt'  # 测试集的所有指标  txt_test_result

    # new txt ver
    ver_id_txt = ver_log_save_path + file_sep+'@' + foldname + '_id_per_person.txt'  #
    ver_label_txt = ver_log_save_path + file_sep+'@' + foldname + '_label_per_person.txt'  #
    ver_pre_txt = ver_log_save_path + file_sep+'@' + foldname + '_pre_per_person.txt'  #
    ver_loss_txt_per_p = ver_log_save_path + file_sep+'@' + foldname + '_loss_per_person.txt'  #

    ver_name_per_block = ver_log_save_path + file_sep+'@' + foldname + '_@_name_per_block.txt'  #
    ver_label_per_block = ver_log_save_path + file_sep+'@' + foldname + '_@_label_per_block.txt'  #
    ver_pre_per_block = ver_log_save_path + file_sep+'@' + foldname + '_@_pre_per_block.txt'  #
    ver_loss_per_block = ver_log_save_path + file_sep+'@' + foldname + '_@_loss_per_block.txt'  #

    # new txt test
    test_id_txt = test_log_save_path + file_sep+'@' + foldname + '_id_per_person.txt'  #
    test_label_txt = test_log_save_path + file_sep+'@' + foldname + '_label_per_person.txt'  #
    test_pre_txt = test_log_save_path + file_sep+'@' + foldname + '_pre_per_person.txt'  #
    test_loss_txt_per_p = test_log_save_path + file_sep+'@' + foldname + '_loss_per_person.txt'  #

    test_name_per_block = test_log_save_path + file_sep+'@' + foldname + '_@_name_per_block.txt'  #
    test_label_per_block = test_log_save_path + file_sep+'@' + foldname + '_@_label_per_block.txt'  #
    test_pre_per_block = test_log_save_path + file_sep+'@' + foldname + '_@_pre_per_block.txt'  #
    test_loss_per_block = test_log_save_path + file_sep+'@' + foldname + '_@_loss_per_block.txt'  #





    # 开始run
    random.shuffle(H5_List_train)
    random.shuffle(H5_List_train)
    index_flag = 0  # 指向训练集列表的index,是更新epoch的依据




    K.set_value(d_model.optimizer.lr, lr_mod_4sgdr(Iter, lr_sgdr))
    # train
    for i in range(max_iter):
        txt_minibatch_loss = open(minibatch_loss_txt, 'a')
        txt_or_loss = open(or_loss_txt, 'a')
        txt_ver_loss = open(ver_loss_txt, 'a')
        txt_test_loss = open(test_loss_txt, 'a')

        txt_lr = open(lr_txt, 'a')
        txt_ver_result = open(verify_result_txt, 'a')
        txt_test_result = open(test_result_txt, 'a')

        Iter = Iter + 1
        subject_num = subject_num + batch_size

        # 读取batch =====================================================
        epoch, index_flag, H5_List_train, data_input_c, \
        label_input_c, filenamelist, labeel = get_batch_from_list(batch_size = batch_size,
                                                                  index_flag = index_flag,
                                                                  filepath_list = H5_List_train,
                                                                  epoch = epoch,
                                                                  label_index = label_index,
                                                                  data_input_shape = data_input_shape,
                                                                  label_shape = label_shape)
        # 读取完成 =======================================================

        pre = d_model.predict_on_batch(data_input_c)
        cost = d_model.train_on_batch(data_input_c, label_input_c,class_weight={0:1,1:1})

        # print the detail of this iter===================================
        if Iter % print_steps == 0:
            # 打印第一组列表
            tb = pt.PrettyTable()
            tb.field_names = [(char_color("task name",50,32)),(char_color("fold",50,32)),(char_color("gpu",50,32)),
                          (char_color("lr",50,32)),(char_color("epoch",50,32)),(char_color("iter",50,32)),
                          (char_color("loss(0~6)",50,32)),(char_color("epoch_subject",50,32))]
            tb.add_row(      [task_name,foldname,os.environ["CUDA_VISIBLE_DEVICES"],
                          K.get_value(d_model.optimizer.lr),epoch,Iter,
                          cost[0],str(subject_num%trainset_num)+'/'+str(trainset_num)])
            tb.align["param_value"] = "l"
            tb.align["param_name"] = "r"
            print(tb)

            # 打印第二组列表
            tb = pt.PrettyTable()
            tb.field_names = [char_color('sub_subject',50,32),char_color('label',50,32),char_color('pre_value',50,32)]
            for ii in range(filenamelist.__len__()):
                sub_subject = filenamelist[ii].split('/')[-1]
                tb.add_row([sub_subject,label_input_c[ii],pre[ii]])
            print(tb)

            # 打印第三组列表
            tb = pt.PrettyTable()
            tb.field_names = [char_color("v_m_acc(iter)",50,35),
                          char_color("v_m_AUC(iter)",50,35),
                          char_color("v_m_sen(iter)",50,35),
                          char_color("v_m_spc(iter)",50,35),
                          char_color("v_m_loss(iter)",50,35),

                          char_color("t_m_acc(iter)",50,36),
                          char_color("t_m_AUC(iter)",50,36),
                          char_color("t_m_sen(iter)",50,36),
                          char_color("t_m_spc(iter)",50,36),
                          char_color("t_m_loss(iter)",50,36)]
            tb.add_row([str(max_acc_verify)+'('+char_color(str(max_acc_verify_iter),50,31)+')',
                    str(max_auc_verify) + '(' + char_color(str(max_auc_verify_iter),50,31) + ')',
                    str(max_sen_verify) + '(' + char_color(str(max_sen_verify_iter),50,31) + ')',
                    str(max_spc_verify) + '(' + char_color(str(max_spc_verify_iter),50,31) + ')',
                    str(min_loss_verify) + '(' + char_color(str(min_loss_verify_iter),50,31) + ')',

                    str(max_acc_test) + '(' + char_color(str(max_acc_test_iter),50,31) + ')',
                    str(max_auc_test) + '(' + char_color(str(max_auc_test_iter),50,31) + ')',
                    str(max_sen_test) + '(' + char_color(str(max_sen_test_iter),50,31) + ')',
                    str(max_spc_test) + '(' + char_color(str(max_spc_test_iter),50,31) + ')',
                    str(min_loss_test) + '(' + char_color(str(min_loss_test_iter),50,31) + ')'])
            print(tb)
            print(Iter,'cost',cost)
        # =============================================================================================

        txt_minibatch_loss.write(str(cost[0]) + '\n')
        txt_lr.write(str(K.get_value(d_model.optimizer.lr)) + '\n')

        # verify & test : only save result , no model will be saved (only use fuc 'test_on_model'or'test_on_model4_subject')
        if Iter >= min_verify_Iters and Iter % verify_Iters_step == 0:
            # 先把保存的flag归零,防止保存多次id之类变量以浪费存储空间
            have_test_flag = True
            # # ver
            vre_result = test_on_model4_subject_new(model=d_model,
                                                test_list=H5_List_ver,
                                                iters=Iter,
                                                data_input_shape=data_input_shape,
                                                label_shape=label_shape,
                                                id_savepath=ver_id_txt,
                                                label_savepath=ver_label_txt,
                                                pre_savepath=ver_pre_txt,
                                                loss_savepath=ver_loss_txt_per_p,

                                                per_block_label_savepath=ver_label_per_block,
                                                per_block_loss_savepath=ver_loss_per_block,
                                                per_block_name_savepath=ver_name_per_block,
                                                per_block_pre_savepath=ver_pre_per_block,

                                                label_index=label_index,
                                                batch_size=test_batchsize,
                                                lossfunc = categorical_crossentropy)
            # save
            txt_ver_result.write(str(Iter) + '@' + str(vre_result) + '\n')
            txt_ver_loss.write(str(Iter) + '@' + str(vre_result[4]) + '\n')

            # test
            test_result = test_on_model4_subject_new(model=d_model,
                                                 test_list=H5_List_test,
                                                 iters=Iter,
                                                 data_input_shape=data_input_shape,
                                                 label_shape=label_shape,
                                                 id_savepath=test_id_txt,
                                                 label_savepath=test_label_txt,
                                                 pre_savepath=test_pre_txt,
                                                 loss_savepath=test_loss_txt_per_p,

                                                 per_block_label_savepath=test_label_per_block,
                                                 per_block_loss_savepath=test_loss_per_block,
                                                 per_block_name_savepath=test_name_per_block,
                                                 per_block_pre_savepath=test_pre_per_block,

                                                 label_index=label_index,
                                                 batch_size=test_batchsize,
                                                 lossfunc = categorical_crossentropy)
            # save
            txt_test_result.write(str(Iter) + '@' + str(test_result) + '\n')
            txt_test_loss.write(str(Iter) + '@' + str(test_result[4]) + '\n')



            # test or_train:这个函数不保存结果到文件,只返回loss
            or_train_result = test_on_model4_subject_new(model=d_model,
                                                test_list=train_list3,
                                                iters=Iter,
                                                data_input_shape=data_input_shape,
                                                label_shape=label_shape,
                                                label_index=label_index,
                                                batch_size=test_batchsize,
                                                or_train_flag=True,
                                                lossfunc=categorical_crossentropy)
            txt_or_loss.write(str(Iter) + '@' + str(or_train_result[4]) + '\n')  # 保存or train 的 loss


            # 更新verify的以供打印的参数 =========================================================================
            if vre_result[0] >= max_acc_verify:
                max_acc_verify_iter = Iter
                max_acc_verify = vre_result[0]
            if vre_result[1] >= max_sen_verify:
                max_sen_verify_iter = Iter
                max_sen_verify = vre_result[1]
            if vre_result[2] >= max_spc_verify:
                max_spc_verify_iter = Iter
                max_spc_verify = vre_result[2]
            if vre_result[3] >= max_auc_verify:
                max_auc_verify_iter = Iter
                max_auc_verify = vre_result[3]
            if vre_result[4] <= min_loss_verify:
                min_loss_verify_iter = Iter
                min_loss_verify = vre_result[4]

            # 更新test的以供打印的参数 =========================================================================
            if test_result[0] >= max_acc_test:
                max_acc_test_iter = Iter
                max_acc_test = test_result[0]
            if test_result[1] >= max_sen_test:
                max_sen_test_iter = Iter
                max_sen_test = test_result[1]
            if test_result[2] >= max_spc_test:
                max_spc_test_iter = Iter
                max_spc_test = test_result[2]
            if test_result[3] >= max_auc_test:
                max_auc_test_iter = Iter
                max_auc_test = test_result[3]
            if test_result[4] <= min_loss_test:
                min_loss_test_iter = Iter
                min_loss_test = test_result[4]

        # 每隔较短时间保存一次模型,防止训练中断
        if Iter % model_save_step == 0:
            # 保存模型
            print('saving model')
            d_model.save(model_save_Path +file_sep +'m_' + 'newest_model.h5')
            print('already save')
        # 每隔较长时间保存一次模型,用来做网络的可视化
        if Iter % model_save_step4_visualization == 0:
            d_model.save(model_save_Path +file_sep+'m_' + str(Iter) + '_model.h5')



        # 转换优化器
        # if Iter == optimizer_switch_point:
        #     print('optimizer_switch_point saving model')
        #     d_model.save(model_save_Path + '/m_' + 'newest_model.h5')
        #     print('optimizer_switch_point saved succeed')
        #     d_model.compile(optimizer=SGD(lr=K.get_value(d_model.optimizer.lr), momentum=0.9), loss='categorical_crossentropy', metrics=[y_t, y_pre, Acc])
        #


        # 学习率策略:抖动循环退火
        # if Iter > optimizer_switch_point:
        #     lr_new = lr_mod(Iter, max_epoch=50, epoch_file_size=trainset_num, batch_size=batch_size, init_lr=trans_lr)
        #     K.set_value(d_model.optimizer.lr, lr_new)

        # 学习率策略:固定补偿衰减
        # if Iter % lr_decay_step == 0:
        #     K.set_value(d_model.optimizer.lr, K.get_value(d_model.optimizer.lr)*decay_rate)

        # 学习率策略:cos循环退火
        K.set_value(d_model.optimizer.lr,lr_mod_4sgdr(Iter, lr_sgdr))




        # 某些变量如测试集id等,之保存一次就够了==============================
        if have_test_flag:
            # new txt ver
            ver_id_txt = None
            ver_label_txt = None
            ver_name_per_block =None
            ver_label_per_block = None
            # new txt test
            test_id_txt = None
            test_label_txt = None
            test_name_per_block = None
            test_label_per_block = None






        # 关闭文件,以供实时查看结果======================================
        txt_minibatch_loss.close()
        txt_ver_result.close()
        txt_test_result.close()
        txt_or_loss.close()
        txt_ver_loss.close()
        txt_test_loss.close()
        txt_lr.close()



