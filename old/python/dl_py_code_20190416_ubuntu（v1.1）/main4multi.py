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
from function import sigmoid_y, pause, verify_on_model, test_on_model, test_on_model4_subject, name2othermode, test_on_model4_subject_4multi
from net import resnetttt, alexnet_jn  # use as :Resnetttt(tuple(data_input_shape+[1]))
from w_resnet import ClassNet,se_ClassNet, multi_input_ClassNet_lk, multi_input_ClassNet
from w_dense_net import se_create_dense_net,create_dense_net
from sklearn import cross_validation,metrics
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
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================

acc_threshold = 0.0  # 准确率阈值(不用设置)
sen_threshold = 0.0  # 敏感度阈值(不用设置)
spc_threshold = 0.0  # 特异度阈值(不用设置)
auc_threshold = 0.0  # auc阈值  (不用设置)



batch_size = 4
max_iter = 200000
min_verify_Iters = 5000   # 开始验证的最小迭代次数
verify_Iters_step = 30  # verify every 'verify_Iters_step' times
data_input_shape = [280, 280, 16]
label_shape = [1]
# label的第一个维度是sample（可以理解为行数）
# step4: parameter set finished


# 分折信息导入
aug_path = r'/data/@data_pnens_zhuanyi_dl/data_aug/a'  # 转移 patient is positive ??
aug_path_othermode = r'/data/@data_pnens_zhuanyi_dl/data_aug/v'  # 门脉期的aug数据

test_path = r'/data/@data_pnens_zhuanyi_dl/data_test/a'
test_path_othermode = r'/data/@data_pnens_zhuanyi_dl/data_test/v' # 门脉期的test数据

folder_path = r'/data/@data_pnens_zhuanyi_dl/CVfold5_1'# 分折信息文件路径

foldname = '5'# 只改动这里即可,无需设定分折文件#============================
batch_name = 'pnens_zhuanyi_resnet_AV'
Result_save_Path = r'/data/XS_Aug_model_result/model_templete/diploma/@pnens_zhuanyi_resnet_av/fold5'#结果储存文件夹========
# model_save_Path = r'/data/XS_Aug_model_result/model_templete/diploma/pnens_zhuanyi_densenet/fold2'#暂时用不到==========

# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================



test_fold_file =folder_path + '/fold_' + foldname + '_test.txt'
verify_fold_file =folder_path + '/fold_' + foldname + '_verify.txt'
train_fold_file =folder_path + '/fold_' + foldname + '_train.txt'

H5_List_test = []
H5_List_verify = []
H5_List_train = []

f = open(test_fold_file, "r")
contents = f.readlines()
for item in contents:
    content = item.strip()
    final_content = test_path + '/' + content
    H5_List_test.append(os.path.join(final_content))
print(H5_List_test)
f.close()

f = open(verify_fold_file, "r")
contents = f.readlines()
for item in contents:
    content = item.strip()
    final_content = test_path + '/' + content
    H5_List_verify.append(os.path.join(final_content))
print(H5_List_verify)
f.close()

f = open(train_fold_file, "r")
contents = f.readlines()
for item in contents:
    content = item.strip()
    final_content = aug_path + '/' + content
    H5_List_train.append(os.path.join(final_content))
print(H5_List_train)
f.close()




trainset_num = len(H5_List_train)
verifset_num = len(H5_List_verify)
testtset_num = len(H5_List_test)
print('train set size :%d'%(trainset_num))
pause()
print('verify set size :%d'%(verifset_num))
pause()
print('test set size :%d'%(testtset_num))
pause()

Num_list_train = list(range(trainset_num))
Num_list_verify = list(range(verifset_num))
Num_list_test = list(range(testtset_num))
# step5: data set prepare finished


# running =====================================================================================================
# train on the train_set by func 'train_on_batch'
# and verify on the verify_set every 'verify_Iters_step' times in the meantime
# the verify mode includes 'thre' and 'maxx' and 'maxx_thre'
# --- mode 'thre' :once verify_value greater than threshold, model save
# --- mode 'maxx' :once current verify_value greater than previous verify_value, model save
# --- mode 'maxx_thre' : Satisfy both conditions (recommended)
# once model is saved, metrics will test on the test_set
if __name__ == "__main__":
    # prepare container
    data_input_1 = np.zeros([batch_size] + data_input_shape + [1], dtype=np.float32)  # net input container
    data_input_2 = np.zeros([batch_size] + data_input_shape + [1], dtype=np.float32)  # net input container
    label_input = np.zeros([batch_size] + label_shape)


    # take memory 4 weight
    #resnet--------------------------------
    #d_model = ClassNet()
    #d_model = se_ClassNet()
    #dense net-------------------------------
    #d_model = create_dense_net(nb_layers=[6,12,24,12],growth_rate=24, nb_filter=64, bottleneck=True, reduction=0.0, dropout_rate=None, subsample_initial_block=True)
    #d_model = se_create_dense_net(nb_layers=[6,12,24,12],growth_rate=24, nb_filter=64, bottleneck=True, reduction=0.0, dropout_rate=None, subsample_initial_block=True)

    d_model = multi_input_ClassNet()
    #d_model = multi_input_ClassNet_lk()


    pause()  # identify
    print(d_model.summary())  # view net
    pause()  # identify


    # extra param initialization
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

    vre_result = [0, 0, 0, 0]  # acc,sen,spc,auc
    test_result = [0, 0, 0, 0]  # acc,sen,spc,auc

    # txt as a log
    loss_txt = Result_save_Path + '/@' + foldname + '_loss.txt'
    verify_result_txt = Result_save_Path + '/@' + foldname + '_ver_result.txt'
    test_result_txt = Result_save_Path + '/@' + foldname + '_test_result.txt'

    verify_acc_txt = Result_save_Path + '/@' + foldname + '_ver_acc.txt'
    verify_AUC_txt = Result_save_Path + '/@' + foldname + '_ver_AUC.txt'
    test_acc_txt = Result_save_Path + '/@' + foldname + '_test_acc.txt'
    test_AUC_txt = Result_save_Path + '/@' + foldname + '_test_AUC.txt'




    txt_s1 = open(loss_txt, 'w')
    txt_s2 = open(verify_result_txt, 'w')
    txt_s3 = open(test_result_txt, 'w')

    txt_s4 = open(verify_acc_txt, 'w')
    txt_s5 = open(verify_AUC_txt, 'w')
    txt_s6 = open(test_acc_txt, 'w')
    txt_s7 = open(test_AUC_txt, 'w')


    # random.shuffle(Num_list_train)
    random.shuffle(H5_List_train)
    random.shuffle(H5_List_train)
    index_flag = 0
    for i in range(max_iter):
        Iter = Iter + 1
        filenamelist = []
        labeel = []
        # start batch input build -----------------------------------------------
        for ii in range(batch_size):
            # 4 a_data: data_input_1--------------------------------
            # read data from h5
            read_name = H5_List_train[index_flag]
            filenamelist.append(read_name)
            H5_file = h5py.File(read_name, 'r')
            batch_x = H5_file['data'][:]
            batch_y = H5_file['label_3'][:]
            H5_file.close()
            labeel.append(batch_y)
            # put data into container
            batch_x_t = np.transpose(batch_x, (1, 2, 0))
            data_input_1[ii, :, :, :, 0] = batch_x_t[:, :, 0:16]
            label_input[ii] = batch_y


            # 4 v_data: data_input_1--------------------------------
            # read data from h5
            read_name_v = name2othermode(aug_path_othermode, read_name, 300)
            H5_file_v = h5py.File(read_name_v, 'r')
            batch_x_v = H5_file_v['data'][:]
            batch_y_v = H5_file_v['label_3'][:]
            H5_file_v.close()
            batch_x_t_v = np.transpose(batch_x_v, (1, 2, 0))
            data_input_2[ii, :, :, :, 0] = batch_x_t_v[:, :, 0:16]


            # index plus 1 and check---------------------------------
            index_flag = index_flag + 1
            if index_flag == trainset_num:
                index_flag = 0
                epoch = epoch +1
        # finish batch input build -----------------------------------------------




        # train on model
        cost = d_model.train_on_batch([data_input_1, data_input_2], label_input)
        # get the pre_result just after train
        pre = d_model.predict_on_batch([data_input_1, data_input_2])
        # print the detail of this iter
        print(batch_name + '=======================================================')
        print('fold:' + foldname)
        print(filenamelist)
        print("\033[0;33;44m·" + '|| Epoch :' + str(epoch) + '|| Iter :' + str(Iter) + '|| Loss :' + str(cost[0]) + "\033[0m")
        print("\033[0;33;46m·" + '|| pre :' + '\n' + str(pre) + "\033[0m")
        print(labeel)
        print("\033[0;33;45m·" + '|| veri_max_acc :' + str(max_acc_verify) + '    iter:' + str(max_acc_verify_iter) + "\033[0m")
        print("\033[0;33;45m·" + '|| test_max_acc :' + str(max_acc_test) + '    iter:' + str(max_acc_test_iter) + "\033[0m")
        print("\033[0;33;45m·" + '|| veri_max_AUC :' + str(max_auc_verify) + '    iter:' + str(max_auc_verify_iter) + "\033[0m")
        print("\033[0;33;45m·" + '|| test_max_AUC :' + str(max_auc_test) + '    iter:' + str(max_auc_test_iter) + "\033[0m")
        # save loss on train set (will save loss both on vre&test set in future)
        txt_s1.write(str(cost[0]) + '\n')
        # verify & test : only save result , no model will be saved (only use fuc 'test_on_model'or'test_on_model4_subject')
        if Iter >= min_verify_Iters and Iter % verify_Iters_step == 0:
            # func 4 verify every 'verify_Iters_step reached' time

            # vre_result = test_on_model(model=d_model, test_list=H5_List_verify, iters=Iter, save_path=Result_save_Path, data_input_shape=data_input_shape,front_name = 'vre')
            vre_result = test_on_model4_subject_4multi(model=d_model, test_list=H5_List_verify, iters=Iter, save_path=Result_save_Path, data_input_shape=data_input_shape,front_name = 'vre',othermode_path = test_path_othermode)
            # save ver_result
            txt_s2.write(str(Iter) + '@' + str(vre_result) + '\n')
            txt_s4.write(str(Iter) + '@' + str(vre_result[0]) + '\n')
            txt_s5.write(str(Iter) + '@' + str(vre_result[3]) + '\n')

            # func 4 test every 'verify_Iters_step reached' time

            # test_result_perfile = test_on_model(model=d_model, test_list=H5_List_test, iters=Iter, save_path=Result_save_Path, data_input_shape=data_input_shape, front_name='test_perfile')
            test_result = test_on_model4_subject_4multi(model=d_model, test_list=H5_List_test, iters=Iter, save_path=Result_save_Path, data_input_shape=data_input_shape, front_name='test',othermode_path = test_path_othermode)
            txt_s3.write(str(Iter) + '@' + str(test_result) + '\n')
            txt_s6.write(str(Iter) + '@' + str(test_result[0]) + '\n')
            txt_s7.write(str(Iter) + '@' + str(test_result[3]) + '\n')


            if vre_result[0] >= max_acc_verify:
                max_acc_verify_iter = Iter
                max_acc_verify = vre_result[0]
            if vre_result[3] >= max_auc_verify:
                max_auc_verify_iter = Iter
                max_auc_verify = vre_result[3]

            if test_result[0] >= max_acc_test:
                max_acc_test_iter = Iter
                max_acc_test = test_result[0]
                # d_model.save(model_save_Path + '/m_' + str(Iter) + '_model.h5')
            if test_result[3] >= max_auc_test:
                max_auc_test_iter = Iter
                max_auc_test = test_result[3]
                # d_model.save(model_save_Path + '/m_' + str(Iter) + '_model.h5')




