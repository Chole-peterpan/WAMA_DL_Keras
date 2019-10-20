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

batch_size = 5
max_iter = 100000
H5_train_path1 = r'/LAOAWNG/diploma/data'  # CD patient is positive
H5_train_path2 = r'/LAOAWNG/diploma/data'
H5_verif_path1 = r'/LAOAWNG/diploma/data'  # +
H5_verif_path2 = r'/LAOAWNG/diploma/data'
H5_testt_path1 = r'/LAOAWNG/diploma/data'  # +
H5_testt_path2 = r'/LAOAWNG/diploma/data'

Result_save_Path = r'/data/XS_Aug_model_result/model_templete/diploma/result'
model_save_Path = r'/data/XS_Aug_model_result/model_templete/diploma/result'

foldname = 'test1'
acc_threshold = 0.0  # 准确率阈值
sen_threshold = 0.0  # 敏感度阈值
spc_threshold = 0.0  # 特异度阈值
auc_threshold = 0.0  # auc阈值
min_verify_Iters = 1   # 开始验证的最小迭代次数
verify_Iters_step = 1    # verify every 'verify_Iters_step' times
data_input_shape = [280, 280, 16]
label_shape = [1]
# label的第一个维度是sample（可以理解为行数）
# step4: parameter set finished


filename_train_posi = os.listdir(H5_train_path1)  # +
filename_train_nega = os.listdir(H5_train_path2)
filename_verif_posi = os.listdir(H5_verif_path1)  # +
filename_verif_nega = os.listdir(H5_verif_path2)
filename_testt_posi = os.listdir(H5_testt_path1)  # +
filename_testt_nega = os.listdir(H5_testt_path2)
# train set ----------------------------------------------------------------------------
H5_List_train = []
for file in filename_train_posi:
    if file.endswith('.h5'):
        H5_List_train.append(os.path.join(H5_train_path1, file))
for file in filename_train_nega:
    if file.endswith('.h5'):
        H5_List_train.append(os.path.join(H5_train_path2, file))
# verify set ----------------------------------------------------------------------------
H5_List_verify = []
for file in filename_verif_posi:
    if file.endswith('.h5'):
        H5_List_verify.append(os.path.join(H5_verif_path1, file))
for file in filename_verif_nega:
    if file.endswith('.h5'):
        H5_List_verify.append(os.path.join(H5_verif_path2, file))
# test set ----------------------------------------------------------------------------
H5_List_test = []
for file in filename_testt_posi:
    if file.endswith('.h5'):
        H5_List_test.append(os.path.join(H5_testt_path1, file))
for file in filename_testt_nega:
    if file.endswith('.h5'):
        H5_List_test.append(os.path.join(H5_testt_path2, file))


print(H5_List_train)
trainset_num = len(H5_List_train)
verifset_num = len(H5_List_verify)
testtset_num = len(H5_List_test)
print('train set size :%d'%(trainset_num))
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
    data_input_c = np.zeros([batch_size] + data_input_shape + [1], dtype=np.float32)  # net input container
    label_input_c = np.zeros([batch_size] + label_shape)
    # take memory 4 weight
    d_model = ClassNet()
    pause()  # identify
    print(d_model.summary())  # view net
    pause()  # identify
    # extra param initialization
    Iter = 0
    epoch = 0
    max_acc_verify = 0
    max_acc_verify_iter = 0
    max_acc_test = 0
    max_acc_test_iter = 0

    max_auc_verify = 0
    max_auc_verify_iter = 0
    max_auc_test = 0
    max_auc_test_iter = 0

    vre_result = [0, 0, 0, 0]  # acc,sen,spc,auc
    test_result = [0, 0, 0, 0]  # acc,sen,spc,auc

    # txt as a log
    loss_txt = Result_save_Path + '/@' + foldname + '_loss.txt'
    verify_result_txt = Result_save_Path + '/@' + foldname + '_ver_acc.txt'
    test_result_txt = Result_save_Path + '/@' + foldname + '_test_acc.txt'
    txt_s1 = open(loss_txt, 'w')
    txt_s2 = open(verify_result_txt, 'w')
    txt_s3 = open(test_result_txt, 'w')

    random.shuffle(Num_list_train)
    index_flag = 0
    for i in range(max_iter):
        Iter = Iter + 1
        # start batch input build -----------------------------------------------
        for ii in range(batch_size):
            # read data from h5
            read_name = H5_List_train[index_flag]
            H5_file = h5py.File(read_name, 'r')
            batch_x = H5_file['data'][:]
            batch_y = H5_file['label_1'][:]
            H5_file.close()
            # put data into container
            batch_x_t = np.transpose(batch_x, (1, 2, 0))
            data_input_c[ii, :, :, :, 0] = batch_x_t[:, :, :]
            label_input_c[ii] = batch_y
            # index plus 1 and check
            index_flag = index_flag + 1
            if index_flag == trainset_num:
                index_flag = 0
                epoch = epoch +1
        # finish batch input build -----------------------------------------------
        # train on model
        cost = d_model.train_on_batch(data_input_c, label_input_c)
        # get the pre_result just after train
        pre = d_model.predict_on_batch(data_input_c)
        # print the detail of this iter
        print('=======================================================')
        print(foldname)
        print("\033[0;33;44m·" + '|| Epoch :' + str(epoch) + '|| Iter :' + str(Iter) + '|| Loss :' + str(cost[0]) + "\033[0m")
        print("\033[0;33;46m·" + '|| pre :' + str(pre) + "\033[0m")
        print("\033[0;33;45m·" + '|| veri_max_acc :' + str(max_acc_verify) + '    iter:' + str(max_acc_verify_iter) + "\033[0m")
        print("\033[0;33;45m·" + '|| test_max_acc :' + str(max_acc_test) + '    iter:' + str(max_acc_test_iter) + "\033[0m")
        print("\033[0;33;45m·" + '|| veri_max_acc :' + str(max_auc_verify) + '    iter:' + str(max_auc_verify_iter) + "\033[0m")
        print("\033[0;33;45m·" + '|| test_max_acc :' + str(max_auc_test) + '    iter:' + str(max_auc_test_iter) + "\033[0m")
        # save loss on train set (will save loss both on vre&test set in future)
        txt_s1.write(str(cost[0]) + '\n')
        # verify & test : only save result , no model will be saved (only use fuc 'test_on_model')
        if Iter >= min_verify_Iters and Iter % verify_Iters_step == 0:
            # func 4 verify every 'verify_Iters_step reached' time
            vre_result = test_on_model(model=d_model, test_list=H5_List_verify, iters=Iter, save_path=Result_save_Path, data_input_shape=data_input_shape)
            # save ver_result
            txt_s2.write(str(Iter) + '@' + str(vre_result) + '\n')
            # func 4 test every 'verify_Iters_step reached' time
            test_result = test_on_model(model=d_model, test_list=H5_List_test, iters=Iter, save_path=Result_save_Path, data_input_shape=data_input_shape)
            txt_s3.write(str(Iter) + '@' + str(test_result) + '\n')

            if vre_result[0] >= max_acc_verify:
                max_acc_verify_iter = Iter
                max_acc_verify = vre_result[0]
            if vre_result[3] >= max_auc_verify:
                max_auc_verify_iter = Iter
                max_auc_verify = vre_result[3]
            if test_result[0] >= max_acc_test:
                max_acc_test_iter = Iter
                max_acc_test = vre_result[0]
                # d_model.save(model_save_Path + '/m_' + str(Iter) + '_model.h5')
            if test_result[3] >= max_auc_verify:
                max_auc_test_iter = Iter
                max_auc_test = vre_result[3]
                # d_model.save(model_save_Path + '/m_' + str(Iter) + '_model.h5')




