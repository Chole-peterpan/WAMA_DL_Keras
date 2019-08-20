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

# os.system('matlab')
config_path = '/data/XS_Aug_model_result/model_templete/testtttttttttttttt/config.txt'

# 从文件读取参数
param_dict = get_param_from_txt(config_path)
pause()

# 提取参数并赋值对应参数到对象
init_lr = param_dict['init_lr']
optimizer_switch_point = param_dict['optm_sw_iter']
model_save_step = param_dict['model_sv_step']
model_save_step4_visualization = param_dict['model_sv_step_4_vis']
os_stage = param_dict['os_stage']
batch_size = param_dict['batch_size']
max_iter = param_dict['max_iter']
min_verify_Iters = param_dict['min_verify_iter']
verify_Iters_step = param_dict['ver_step']
data_input_shape = (param_dict['data_input_shape'])
label_index = param_dict['label_index']
label_shape = (param_dict['label_shape'])
aug_path = param_dict['aug_subject_path']
or_path = param_dict['or_subject_path']
folder_path = param_dict['folder_path']
Result_save_Path = param_dict['result_save_path']
foldname = param_dict['fold_name']
task_name = param_dict['task_name']

if os_stage == "W":
    file_sep = r"\\"
elif os_stage == "L":
    file_sep = r'/'
else:
    file_sep = r'/'

# mkdir log和model两个文件夹,并将参数文件备份到log文件夹中
log_save_Path = Result_save_Path + file_sep[0] + 'log'
model_save_Path = Result_save_Path + file_sep[0] + 'model'
os.system('mkdir '+log_save_Path)
os.system('mkdir '+model_save_Path)
os.system('cp '+config_path+' '+log_save_Path)

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = param_dict['GPU_index']
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# 读取分折信息
test_fold_file = folder_path + file_sep[0] + 'fold_' + foldname + '_test.txt'
verify_fold_file = folder_path + file_sep[0] + 'fold_' + foldname + '_verify.txt'
train_fold_file = folder_path + file_sep[0] + 'fold_' + foldname + '_train.txt'
or_train_fold_file = folder_path + file_sep[0] + 'fold_' + foldname + '_ortrain.txt'  # 未扩增的训练集

H5_List_test = []
H5_List_verify = []
H5_List_train = []
H5_List_or_train = []

f = open(test_fold_file, "r")
contents = f.readlines()
for item in contents:
    content = item.strip()
    final_content = or_path + file_sep[0] + content
    H5_List_test.append(os.path.join(final_content))
print(H5_List_test)
f.close()

f = open(verify_fold_file, "r")
contents = f.readlines()
for item in contents:
    content = item.strip()
    final_content = or_path + file_sep[0] + content
    H5_List_verify.append(os.path.join(final_content))
print(H5_List_verify)
f.close()

f = open(train_fold_file, "r")
contents = f.readlines()
for item in contents:
    content = item.strip()
    final_content = aug_path + file_sep[0] + content
    H5_List_train.append(os.path.join(final_content))
print(H5_List_train)
f.close()

f = open(or_train_fold_file, "r")
contents = f.readlines()
for item in contents:
    content = item.strip()
    final_content = or_path + file_sep[0] + content
    H5_List_or_train.append(os.path.join(final_content))
print(H5_List_or_train)
f.close()



# 静脉期的id号比动脉期大300,所以需要调整下

for i in range(H5_List_test.__len__()):
    tmp = H5_List_test[i]
    H5_List_test[i] = name2othermode(or_path, tmp, 300)

for i in range(H5_List_verify.__len__()):
    tmp = H5_List_verify[i]
    H5_List_verify[i] = name2othermode(or_path, tmp, 300)

for i in range(H5_List_train.__len__()):
    tmp = H5_List_train[i]
    H5_List_train[i] = name2othermode(aug_path, tmp, 300)

for i in range(H5_List_or_train.__len__()):
    tmp = H5_List_or_train[i]
    H5_List_or_train[i] = name2othermode(or_path, tmp, 300)




trainset_num = len(H5_List_train)
verifset_num = len(H5_List_verify)
testtset_num = len(H5_List_test)
or_train_num = len(H5_List_or_train)
print('train set size :%d' % trainset_num)
pause()
print('verify set size :%d' % verifset_num)
pause()
print('test set size :%d' % testtset_num)
pause()
print('or_train set size :%d' % or_train_num)
pause()


# 构建各个集合中样本文件索引的list,以供接下来使用
Num_list_train = list(range(trainset_num))
Num_list_verify = list(range(verifset_num))
Num_list_test = list(range(testtset_num))
Num_list_or_train = list(range(or_train_num))


if __name__ == "__main__":
    # prepare container:准备网络输入输出容器
    data_input_c = np.zeros([batch_size] + data_input_shape + [1], dtype=np.float32)  # net input container
    label_input_c = np.zeros([batch_size] + label_shape)
    # 构建网络并compile
    d_model = resnet_or(use_bias_flag=True,classes=2)
    d_model.compile(optimizer=adam(lr=init_lr), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    # pause()  # identify
    print(d_model.summary())  # view net
    # pause()  # identify
    # extra param initialization:初始化一些用来记录和显示的参数
    Iter = 0
    epoch = 0
    max_acc_verify = 0.6
    max_acc_verify_iter = 0
    max_acc_test = 0.6
    max_acc_test_iter = 0

    max_auc_verify = 0.6
    max_auc_verify_iter = 0
    max_auc_test = 0.6
    max_auc_test_iter = 0

    max_spc_verify = 0.6
    max_spc_verify_iter = 0
    max_spc_test = 0.6
    max_spc_test_iter = 0

    max_sen_verify = 0.6
    max_sen_verify_iter = 0
    max_sen_test = 0.6
    max_sen_test_iter = 0

    min_loss_verify = 50
    min_loss_verify_iter = 0
    min_loss_test = 50
    min_loss_test_iter = 0


    vre_result = [0, 0, 0, 0]  # acc,sen,spc,auc
    test_result = [0, 0, 0, 0]  # acc,sen,spc,auc

    # txt as a log
    minibatch_loss_txt = log_save_Path + file_sep[0]+'@' + foldname + '_loss.txt'  # minibatch上的loss  txt_minibatch_loss
    or_loss_txt = log_save_Path + file_sep[0]+'@' + foldname + '_or_loss.txt'  # 在原未扩增训练集上的loss  txt_or_loss
    ver_loss_txt = log_save_Path + file_sep[0]+'@' + foldname + '_ver_loss.txt'  # 在验证集上的loss  txt_ver_loss
    test_loss_txt = log_save_Path + file_sep[0]+'@' + foldname + '_test_loss.txt'  # 在测试集上的loss  txt_test_loss

    verify_result_txt = log_save_Path + file_sep[0]+'@' + foldname + '_ver_result.txt'  # 验证集的所有指标  txt_ver_result
    test_result_txt = log_save_Path + file_sep[0]+'@' + foldname + '_test_result.txt'  # 测试集的所有指标  txt_test_result

    lr_txt = log_save_Path + file_sep[0]+'@' + foldname + '_lr.txt'  # 学习率曲线  txt_s11

    # new txt
    ver_id_txt = log_save_Path + file_sep[0]+'@' + foldname + '_ver_id.txt'  #
    ver_label_txt = log_save_Path + file_sep[0]+'@' + foldname + '_ver_label.txt'  #
    ver_pre_txt = log_save_Path + file_sep[0]+'@' + foldname + '_ver_pre.txt'  #

    test_id_txt = log_save_Path + file_sep[0]+'@' + foldname + '_test_id.txt'  #
    test_label_txt = log_save_Path + file_sep[0]+'@' + foldname + '_test_label.txt'  #
    test_pre_txt = log_save_Path + file_sep[0]+'@' + foldname + '_test_pre.txt'  #

    # 开始run
    random.shuffle(H5_List_train)
    random.shuffle(H5_List_train)
    index_flag = 0
    for i in range(max_iter):
        txt_minibatch_loss = open(minibatch_loss_txt, 'a')
        txt_or_loss = open(or_loss_txt, 'a')
        txt_ver_loss = open(ver_loss_txt, 'a')
        txt_test_loss = open(test_loss_txt, 'a')

        txt_lr = open(lr_txt, 'a')
        txt_ver_result = open(verify_result_txt, 'a')
        txt_test_result = open(test_result_txt, 'a')

        Iter = Iter + 1
        filenamelist = []
        labeel = []
        # start batch input build -----------------------------------------------
        for ii in range(batch_size):
            # read data from h5
            read_name = H5_List_train[index_flag]
            print(read_name)
            filenamelist.append(read_name)
            H5_file = h5py.File(read_name, 'r')
            batch_x = H5_file['data'][:]
            batch_y = H5_file[label_index][:]
            H5_file.close()
            labeel.append(batch_y)
            # put data into container
            batch_x_t = np.transpose(batch_x, (1, 2, 0))
            data_input_c[ii, :, :, :, 0] = batch_x_t[:, :, 0:16]
            label_input_c[ii] = batch_y#有点微妙啊这个地方,有点二元数组的感觉
            # index plus 1 and check
            index_flag = index_flag + 1
            if index_flag == trainset_num:
                index_flag = 0
                epoch = epoch +1
                random.shuffle(H5_List_train)  # new
        # finish batch input build -----------------------------------------------
        # train on model
        cost = d_model.train_on_batch(data_input_c, label_input_c)
        pre = d_model.predict_on_batch(data_input_c)


        # print the detail of this iter
        tb = pt.PrettyTable()
        tb.field_names = [(char_color("task name",50,32)),(char_color("fold",50,32)),(char_color("gpu",50,32)),
                          (char_color("lr",50,32)),(char_color("epoch",50,32)),(char_color("iter",50,32)),
                          (char_color("loss",50,32))]
        tb.add_row(      [task_name,foldname,os.environ["CUDA_VISIBLE_DEVICES"],
                          K.get_value(d_model.optimizer.lr),epoch,Iter,
                          cost[0]])
        tb.align["param_value"] = "l"
        tb.align["param_name"] = "r"
        print(tb)




        print(filenamelist)

        print("\033[0;33;46m·" + '|| pre :' + "\033[0m")
        print("\033[0;33;46m·" + str((pre)) + "\033[0m")
        print("\033[0;33;46m·" + '|| label :' + "\033[0m")
        print("\033[0;33;46m·" + str(label_input_c) + "\033[0m")

        print("\033[0;33;45m·" + '|| veri_max_acc :' + str(max_acc_verify) + '    iter:' + str(max_acc_verify_iter) + "\033[0m")
        print("\033[0;33;45m·" + '|| test_max_acc :' + str(max_acc_test) + '    iter:' + str(max_acc_test_iter) + "\033[0m")
        print("\033[0;33;44m·" + '|| veri_max_AUC :' + str(max_auc_verify) + '    iter:' + str(max_auc_verify_iter) + "\033[0m")
        print("\033[0;33;44m·" + '|| test_max_AUC :' + str(max_auc_test) + '    iter:' + str(max_auc_test_iter) + "\033[0m")
        print("\033[0;33;40m·" + '|| veri_max_sen :' + str(max_sen_verify) + '    iter:' + str(max_sen_verify_iter) + "\033[0m")
        print("\033[0;33;40m·" + '|| test_max_sen :' + str(max_sen_test) + '    iter:' + str(max_sen_test_iter) + "\033[0m")
        print("\033[0;33;42m·" + '|| veri_max_spc :' + str(max_spc_verify) + '    iter:' + str(max_spc_verify_iter) + "\033[0m")
        print("\033[0;33;42m·" + '|| test_max_spc :' + str(max_spc_test) + '    iter:' + str(max_spc_test_iter) + "\033[0m")

        print("\033[0;33;40m·" + '|| veri_min_loss :' + str(min_loss_verify) + '    iter:' + str(min_loss_verify_iter) + "\033[0m")
        print("\033[0;33;40m·" + '|| test_min_loss :' + str(min_loss_test) + '    iter:' + str(min_loss_test_iter) + "\033[0m")

        # save loss on train set (will save loss both on vre&test set in future)
        txt_minibatch_loss.write(str(cost[0]) + '\n')
        # save the current lr
        txt_lr.write(str(K.get_value(d_model.optimizer.lr)) + '\n')

        # verify & test : only save result , no model will be saved (only use fuc 'test_on_model'or'test_on_model4_subject')
        if Iter >= min_verify_Iters and Iter % verify_Iters_step == 0:
            # ver
            vre_result = test_on_model4_subject(model=d_model,
                                                test_list=H5_List_verify,
                                                iters=Iter,
                                                data_input_shape=data_input_shape,
                                                label_shape=label_shape,
                                                id_savepath=ver_id_txt,
                                                label_savepath=ver_label_txt,
                                                pre_savepath=ver_pre_txt,
                                                label_index=label_index)
            # save
            txt_ver_result.write(str(Iter) + '@' + str(vre_result) + '\n')
            txt_ver_loss.write(str(Iter) + '@' + str(vre_result[4]) + '\n')

            # test
            test_result = test_on_model4_subject(model=d_model,
                                                test_list=H5_List_test,
                                                iters=Iter,
                                                data_input_shape=data_input_shape,
                                                label_shape=label_shape,
                                                id_savepath=test_id_txt,
                                                label_savepath=test_label_txt,
                                                pre_savepath=test_pre_txt,
                                                label_index=label_index)
            # save
            txt_test_result.write(str(Iter) + '@' + str(test_result) + '\n')
            txt_test_loss.write(str(Iter) + '@' + str(test_result[4]) + '\n')



            # test or_train:这个函数不保存结果到文件,只返回loss
            or_train_result = test_on_model4_subject4_or_train(model=d_model,
                                                               test_list=H5_List_or_train,
                                                               data_input_shape=data_input_shape,
                                                               label_shape=label_shape,
                                                               label_index=label_index)

            txt_or_loss.write(str(Iter) + '@' + str(or_train_result) + '\n')  # 保存or train 的 loss


            # 更新verify的log以供打印 =========================================================================
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

            # 更新test的log以供打印 =========================================================================
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




# 保存尽量新模型,防止训练中断
        if Iter % model_save_step == 0:
            d_model.save(model_save_Path +file_sep +'m_' + 'newest_model.h5')

# 每隔较长时间保存一次模型,用来做网络的可视化

        if Iter % model_save_step4_visualization == 0:
            d_model.save(model_save_Path +file_sep+'m_' + str(Iter) + '_model.h5')

# 优化策略:优化器变更以及学习率调整=========================================================================================
#       #开始的时候我们用的是adam,所以下面代码分为两部分,一部分是切换adam到sgd,另外一部分负责切换之后更新学习率
        #如果到达转换点,那么就开始转换,以防万一,先保存权重,之后重新编译模型,之后加载权重
        if Iter == optimizer_switch_point:
            d_model.save(model_save_Path + '/m_' + 'newest_model.h5')
            lr_new = lr_mod(Iter, max_epoch=50, epoch_file_size=trainset_num, batch_size=batch_size, init_lr=init_lr)
            d_model.compile(optimizer=SGD(lr=lr_new, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
            d_model.load_weights(model_save_Path + '/m_' + 'newest_model.h5')

        if Iter > optimizer_switch_point:
            #batch_num_perepoch = or_train_num // batch_size  # 每个epoch包含的迭代次数,也即batch的个数
            lr_new = lr_mod(Iter, max_epoch=50, epoch_file_size=trainset_num, batch_size=batch_size, init_lr=init_lr)
            K.set_value(d_model.optimizer.lr, lr_new)



# 关闭文件,以供实时查看结果
        txt_minibatch_loss.close()
        txt_ver_result.close()
        txt_test_result.close()
        txt_or_loss.close()
        txt_ver_loss.close()
        txt_test_loss.close()
        txt_lr.close()





