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
from w_resnet import resnet, resnet_nobn, se_resnet
from w_dense_net import dense_net,se_dense_net
from w_resnext import  resnext, resnext_or, se_resnext
from w_vggnet import vgg16_w_3d, vgg16_w_3d_gb
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss

import keras.backend as K

# step2: import extra model finished


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# --step3: GPU configuration finished



#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
# --step3.1: (mode 1 use cpu)or can use cpu forcibly with above code
# and should annotation 4 gpu configuration


# 参数设置===============(手动设置部分)===================================================================================
#acc_threshold = 0.0  # 准确率阈值(不用设置)
#sen_threshold = 0.0  # 敏感度阈值(不用设置)
#spc_threshold = 0.0  # 特异度阈值(不用设置)
#auc_threshold = 0.0  # auc阈值  (不用设置)


first_lr = 2e-6
optimizer_switch_point = 1800 # 优化器从adam转换到sgd的迭代数(时间点),一般是1到3个epoch,这个自己计算


model_save_step = 500 #每迭代model_save_step个batch,保存一次模型
model_save_step4_visualization = 5000 #每迭代model_save_step4_visualization个batch,保存一次模型用来做可视化,一般来说这个参数要比model_save_step大一些

os_stage = "L"  # W:windows or L:linux
batch_size = 6
max_iter = 200000
min_verify_Iters = 10   # 开始验证的最小迭代次数:batch*diedaishu = epoch
verify_Iters_step = 50  # verify every 'verify_Iters_step' times
data_input_shape = [280, 280, 16]
label_index = 'label3'#label3
label_shape = [2]
# label的第一个维度是sample（可以理解为行数）
# step4: parameter set finished

# 分折信息导入
aug_path = r'/data/@data_liaoxiao/3aug_10000'  # CD patient is positive
test_path = r'/data/@data_liaoxiao/4test'
folder_path = r'/data/@data_liaoxiao/@folder_5_10000new'# 分折信息文件路径

Result_save_Path = r'/data/XS_Aug_model_result/model_templete/diploma_real/test22'#============================#!!!!!! 保存路径保存路径保存路径保存路径保存路径
model_save_Path = Result_save_Path#保存模型l路径,和结果存放路径相同
foldname = '2'# 只改动这里即可,无需设定分折文件#============================
task_name = 'test22'#!!!!!!



#======================================================================================================================
if os_stage == "W":
    file_sep = r"\\"
elif os_stage == "L":
    file_sep = r'/'
else:
    file_sep = r'/'


test_fold_file = folder_path + file_sep[0] + 'fold_' + foldname + '_test.txt'
verify_fold_file = folder_path + file_sep[0] + 'fold_' + foldname + '_verify.txt'
train_fold_file = folder_path + file_sep[0] + 'fold_' + foldname + '_train.txt'
or_train_fold_file = folder_path + file_sep[0] + 'fold_' + foldname + '_ortrain.txt'  # 使用test组里的数据（即未扩增的数据）


H5_List_test = []
H5_List_verify = []
H5_List_train = []
H5_List_or_train = []


f = open(test_fold_file, "r")
contents = f.readlines()
for item in contents:
    content = item.strip()
    final_content = test_path + file_sep[0] + content
    H5_List_test.append(os.path.join(final_content))
print(H5_List_test)
f.close()

f = open(verify_fold_file, "r")
contents = f.readlines()
for item in contents:
    content = item.strip()
    final_content = test_path + file_sep[0] + content
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
    final_content = test_path + file_sep[0] + content
    H5_List_or_train.append(os.path.join(final_content))
print(H5_List_or_train)
f.close()




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


# 下面参数没用，算着玩
Num_list_train = list(range(trainset_num))
Num_list_verify = list(range(verifset_num))
Num_list_test = list(range(testtset_num))
Num_list_or_train = list(range(or_train_num))


if __name__ == "__main__":


    #实验20：10000,di 2 zhe
    d_model = resnet(use_bias_flag=True,classes=2,weight_decay=0.)
    d_model.compile(optimizer=adam(lr=2e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])



    lr_txt = Result_save_Path + file_sep[0]+'@' + foldname + '_lr.txt'  # 学习率曲线  txt_s11

    Iter=0
    for i in range(max_iter):
        Iter= Iter+1
        txt_s11 = open(lr_txt, 'a')
        txt_s11.write(str(K.get_value(d_model.optimizer.lr)) + '\n')


        if Iter == optimizer_switch_point:
            lr_new = lr_mod(Iter, max_epoch=50, epoch_file_size=trainset_num, batch_size=batch_size, init_lr=first_lr)
            d_model.compile(optimizer=SGD(lr=lr_new, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])

        if Iter > optimizer_switch_point:
            #batch_num_perepoch = or_train_num // batch_size  # 每个epoch包含的迭代次数,也即batch的个数
            lr_new = lr_mod(Iter, max_epoch=50, epoch_file_size=trainset_num, batch_size=batch_size, init_lr=first_lr)
            K.set_value(d_model.optimizer.lr, lr_new)

# 关闭文件,以供实时查看结果
        txt_s11.close()






