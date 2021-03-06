from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# step1: import model finished
from function import pause, test_on_model4_subject, test_on_model4_subject4_or_train, lr_mod, name2othermode
from w_dualpathnet import dual_path_net
from w_resnet import resnet_nobn, se_resnet, resnet_or
from w_dense_net import dense_net,se_dense_net
from w_resnext import  resnext, resnext_or, se_resnext
from w_vggnet import vgg16_w_3d, vgg16_w_3d_gb
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss

import keras.backend as K

# step2: import extra model finished


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
min_verify_Iters = 4000   # 开始验证的最小迭代次数:batch*diedaishu = epoch
verify_Iters_step = 50  # verify every 'verify_Iters_step' times
data_input_shape = [280, 280, 16]
label_index = 'label3'#label3
label_shape = [2]
# label的第一个维度是sample（可以理解为行数）
# step4: parameter set finished

# 分折信息导入
# 注意,这里是跑静脉期的,所以是静脉期限的数据路径
aug_path = r'/data/@data_pnens_zhuanyi_dl/data_aug/v'  # CD patient is positive
test_path = r'/data/@data_pnens_zhuanyi_dl/data_test/v'
folder_path = r'/data/@data_pnens_zhuanyi_dl/CVfold5_newnew'# 分折信息文件路径

Result_save_Path = r'/data/XS_Aug_model_result/model_templete/recurrent/pnens_zhuanyi_resnet_v_new(fuxk)/fold6'#============================#!!!!!! 保存路径保存路径保存路径保存路径保存路径
model_save_Path = Result_save_Path#保存模型l路径,和结果存放路径相同
foldname = '5'# 只改动这里即可,无需设定分折文件#============================
task_name = 'recurrent fold5'#!!!!!!



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



# 静脉期的id号比动脉期大300,所以需要调整下

for i in range(H5_List_test.__len__()):
    tmp = H5_List_test[i]
    H5_List_test[i] = name2othermode(test_path, tmp, 300)

for i in range(H5_List_verify.__len__()):
    tmp = H5_List_verify[i]
    H5_List_verify[i] = name2othermode(test_path, tmp, 300)

for i in range(H5_List_train.__len__()):
    tmp = H5_List_train[i]
    H5_List_train[i] = name2othermode(aug_path, tmp, 300)

for i in range(H5_List_or_train.__len__()):
    tmp = H5_List_or_train[i]
    H5_List_or_train[i] = name2othermode(test_path, tmp, 300)




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

    #resnet=============================================================================
    #d_model = resnet()
    #d_model = resnet_nobn()
    #d_model = se_resnet()
    # resneXt=============================================================================
    # dense net=============================================================================
    #d_model = dense_net(nb_layers=[6,12,24,12],growth_rate=24, nb_filter=64, bottleneck=True, reduction=0.0, dropout_rate=None, subsample_initial_block=True)
    #d_model = se_dense_net(nb_layers=[6,12,24,12],growth_rate=24, nb_filter=64, bottleneck=True, reduction=0.0, dropout_rate=None, subsample_initial_block=True)
    # dual path net=============================================================================
    #d_model = dual_path_net(initial_conv_filters=64, filter_increment=[16, 32, 24, 128], depth=[3, 4, 20, 3], cardinality=32, width=3, pooling='max-avg')
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    #d_model.compile(optimizer=adam(lr=1e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])

    #test log===================================================================================
    #实验记录
    #实验1：10000,di 1 zhe
    #d_model = resnet(use_bias_flag=True)
    #d_model.compile(optimizer=adam(lr=1e-5), loss=EuiLoss, metrics=[y_t, y_pre, Acc])

    #实验2：10000,di 1 zhe
    #d_model = resnet(use_bias_flag=False)
    #d_model.compile(optimizer=adam(lr=1e-5), loss=EuiLoss, metrics=[y_t, y_pre, Acc])

    #实验3：没有bn层,貌似不收敛,替换为有bn层的加SGD,10000,di 1 zhe
    #d_model = resnet(use_bias_flag=True)
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])

    #实验4：10000,di 2 zhe
    #d_model = resnet(use_bias_flag=True)
    #d_model.compile(optimizer=adam(lr=1e-5), loss=EuiLoss, metrics=[y_t, y_pre, Acc])

    #实验5：10000,di 2 zhe
    #d_model = resnet(use_bias_flag=True)
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])

    #实验6：10000,di 1 zhe
    #d_model = dual_path_net(initial_conv_filters = 64, filter_increment = [16, 32, 24, 128], depth = [3, 4, 6, 3], cardinality=16, width=3, pooling='max-avg',bias_flag=True)
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])#貌似card和width才是控制网络大小的重要参数啊

    #实验7：10000,di 1 zhe
    #d_model = resnext()
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])

    #实验8：10000,di 3 zhe
    #d_model = resnet(use_bias_flag=True)
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])

    #实验9：10000,di 4 zhe
    #d_model = resnet(use_bias_flag=True)
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])


    #实验10：10000,di 5 zhe
    #d_model = resnet(use_bias_flag=True)
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])


    #实验11：10000,di 2 zhe
    #d_model = resnet(use_bias_flag=True,classes=2)
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss='binary_crossentropy', metrics=[y_t, y_pre, Acc])



    #实验12：10000,di 2 zhe
    #d_model = resnext(classes=2,use_bias_flag=True)
    #d_model.compile(optimizer=SGD(lr=1e-5, momentum=0.9), loss='binary_crossentropy', metrics=[y_t, y_pre, Acc])


    #实验13：10000,di 2 zhe
    #d_model = dense_net(nb_layers=[6,12,24,12],growth_rate=24, nb_filter=64, bottleneck=True, reduction=0.0, dropout_rate=None, subsample_initial_block=True, classes = 2)
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss='binary_crossentropy', metrics=[y_t, y_pre, Acc])



    #实验14：10000,di 2 zhe
    #d_model = resnet(use_bias_flag=True,classes=2)
    #d_model.compile(optimizer=SGD(lr=1e-5, momentum=0.9), loss='binary_crossentropy', metrics=[y_t, y_pre, Acc])



    #实验15：10000,di 2 zhe
    #d_model = resnet(use_bias_flag=True,classes=2)
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])



    #实验16：10000,di 2 zhe
    #d_model = resnext(classes=2,use_bias_flag=True)
    #d_model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='binary_crossentropy', metrics=[y_t, y_pre, Acc])



    #实验17：10000,di 2 zhe
    #d_model = dense_net(nb_layers=[6,12,24,12],growth_rate=24, nb_filter=64, bottleneck=True, reduction=0.0, dropout_rate=None, subsample_initial_block=True, classes = 2)
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])



    #实验18：10000,di 2 zhe
    #d_model = resnext(classes=2,use_bias_flag=True)
    #d_model.compile(optimizer=SGD(lr=1e-5, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])


    #实验19：10000,di 2 zhe
    #d_model = resnet(use_bias_flag=True,classes=2,weight_decay=0.)
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    #d_model.load_weights(filepath='/data/XS_Aug_model_result/model_templete/qianyi/model_qianyi_new/3RECURRENT/m_30000_model.h5', by_name=True)



    #实验20：10000,di 2 zhe
    #d_model = resnet(use_bias_flag=True,classes=2,weight_decay=1e-3)
    #d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    #d_model.load_weights(filepath='/data/XS_Aug_model_result/model_templete/qianyi/model_qianyi_new/3RECURRENT/m_30000_model.h5', by_name=True)




    #实验21：10000,di 2 zhe
    #d_model = resnet(use_bias_flag=True,classes=2,weight_decay=0.)
    #d_model.compile(optimizer=adam(lr=1e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    #d_model.load_weights(filepath='/data/XS_Aug_model_result/model_templete/qianyi/model_qianyi_new/3RECURRENT/m_30000_model.h5', by_name=True)





    #实验22：10000,di 2 zhe
    #d_model = resnet1(use_bias_flag=True,classes=2)
    #d_model.compile(optimizer=adam(lr=2e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    #d_model.load_weights(filepath='/data/XS_Aug_model_result/model_templete/qianyi/model_qianyi_new/3RECURRENT/m_30000_model.h5', by_name=True)




    #实验23：10000,di 2 zhe
    #d_model = resnet1(use_bias_flag=True,classes=2)
    #d_model.compile(optimizer=SGD(lr=2e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    #d_model.load_weights(filepath='/data/XS_Aug_model_result/model_templete/qianyi/model_qianyi_new/3RECURRENT/m_30000_model.h5', by_name=True)

    #实验24：10000,di 2 zhe
    #d_model = resnet(use_bias_flag=True,classes=2,weight_decay=1e-3)
    #d_model.compile(optimizer=SGD(lr=2e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    #d_model.load_weights(filepath='/data/XS_Aug_model_result/model_templete/qianyi/model_qianyi_new/3RECURRENT/m_30000_model.h5', by_name=True)


    #实验32：10000,di 2 zhe
    # d_model = resnext(classes=2,use_bias_flag=True)
    # d_model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])


    #实验33：10000,di 2 zhe
    # d_model = resnet_or(use_bias_flag=True,classes=2)
    # d_model.compile(optimizer=adam(lr=2e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])

    #实验38：10000,di 3 zhe
    # d_model = resnet_or(use_bias_flag=True,classes=2)
    # d_model.compile(optimizer=adam(lr=2e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    #
    #实验39：10000,di 4 zhe
    # d_model = resnet_or(use_bias_flag=True,classes=2)
    # d_model.compile(optimizer=adam(lr=2e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    #
    #实验40：10000,di 5 zhe
    # d_model = resnet_or(use_bias_flag=True,classes=2)
    # d_model.compile(optimizer=adam(lr=2e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    #
    # #实验41：10000,di 1 zhe
    # d_model = resnet_or(use_bias_flag=True,classes=2)
    # d_model.compile(optimizer=adam(lr=2e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    #

    #
    # #实验:重新跑复发的,第一折
    # d_model = resnet_or(use_bias_flag=True,classes=2)
    # d_model.compile(optimizer=adam(lr=2e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    # #===================================================================================


    #实验:重新跑复发的,静脉期的
    d_model = resnet_or(use_bias_flag=True,classes=2)
    d_model.compile(optimizer=adam(lr=2e-6), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
    #===================================================================================



    pause()  # identify
    print(d_model.summary())  # view net
    #print(model.summary())  # view net
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
    minibatch_loss_txt = Result_save_Path + file_sep[0]+'@' + foldname + '_loss.txt'  # minibatch上的loss  txt_s1

    verify_result_txt = Result_save_Path + file_sep[0]+'@' + foldname + '_ver_result.txt'  # 验证集的所有指标  txt_s2
    test_result_txt = Result_save_Path + file_sep[0]+'@' + foldname + '_test_result.txt'  # 测试集的所有指标  txt_s3

    verify_acc_txt = Result_save_Path + file_sep[0]+'@' + foldname + '_ver_acc.txt'  # 验证集的acc  txt_s4
    verify_AUC_txt = Result_save_Path + file_sep[0]+'@' + foldname + '_ver_AUC.txt'  # 验证集的AUC  txt_s5
    test_acc_txt = Result_save_Path + file_sep[0]+'@' + foldname + '_test_acc.txt'  # 测试集的acc  txt_s6
    test_AUC_txt = Result_save_Path + file_sep[0]+'@' + foldname + '_test_AUC.txt'  # 测试集的AUC  txt_s7

    or_loss_txt = Result_save_Path + file_sep[0]+'@' + foldname + '_or_loss.txt'  # 在原未扩增训练集上的loss  txt_s8
    ver_loss_txt = Result_save_Path + file_sep[0]+'@' + foldname + '_ver_loss.txt'  # 在验证集上的loss  txt_s9
    test_loss_txt = Result_save_Path + file_sep[0]+'@' + foldname + '_test_loss.txt'  # 在测试集上的loss  txt_s10

    lr_txt = Result_save_Path + file_sep[0]+'@' + foldname + '_lr.txt'  # 学习率曲线  txt_s11







    # random.shuffle(Num_list_train)
    random.shuffle(H5_List_train)
    random.shuffle(H5_List_train)
    index_flag = 0
    for i in range(max_iter):
        txt_s1 = open(minibatch_loss_txt, 'a')
        txt_s2 = open(verify_result_txt, 'a')
        txt_s3 = open(test_result_txt, 'a')

        txt_s4 = open(verify_acc_txt, 'a')
        txt_s5 = open(verify_AUC_txt, 'a')
        txt_s6 = open(test_acc_txt, 'a')
        txt_s7 = open(test_AUC_txt, 'a')

        txt_s8 = open(or_loss_txt, 'a')
        txt_s9 = open(ver_loss_txt, 'a')
        txt_s10 = open(test_loss_txt, 'a')

        txt_s11 = open(lr_txt, 'a')



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
        # test on model :相当于只计算loss，不更新梯度
        # cost_tmp = d_model.test_on_batch(data_input_c, label_input_c)
        # get the pre_result just after train
        pre = d_model.predict_on_batch(data_input_c)
        # print the detail of this iter
        print(task_name + '=======================================================')
        print('fold:' + foldname)
        print('gpu:'+os.environ["CUDA_VISIBLE_DEVICES"])
        print(filenamelist)
        print('lr: ', K.get_value(d_model.optimizer.lr))
        print("\033[0;33;44m·" + '|| Epoch :' + str(epoch) + '|| Iter :' + str(Iter) + '|| Loss :' + str(cost) + "\033[0m")
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
        txt_s1.write(str(cost[0]) + '\n')
        # save the current lr
        txt_s11.write(str(K.get_value(d_model.optimizer.lr)) + '\n')

        # verify & test : only save result , no model will be saved (only use fuc 'test_on_model'or'test_on_model4_subject')
        if Iter >= min_verify_Iters and Iter % verify_Iters_step == 0:

            # func 4 verify every 'verify_Iters_step reached' time ====================================
            # vre_result = test_on_model(model=d_model, test_list=H5_List_verify, iters=Iter, save_path=Result_save_Path, data_input_shape=data_input_shape,front_name = 'vre')
            vre_result = test_on_model4_subject(model=d_model, test_list=H5_List_verify, iters=Iter, save_path=Result_save_Path, data_input_shape=data_input_shape, label_shape=label_shape, front_name='vre', file_sep=file_sep,label_index=label_index)
            # save ver_result
            txt_s2.write(str(Iter) + '@' + str(vre_result) + '\n')
            txt_s4.write(str(Iter) + '@' + str(vre_result[0]) + '\n')
            txt_s5.write(str(Iter) + '@' + str(vre_result[3]) + '\n')
            txt_s9.write(str(Iter) + '@' + str(vre_result[4]) + '\n')

            # func 4 test every 'verify_Iters_step reached' time ====================================
            # test_result_perfile = test_on_model(model=d_model, test_list=H5_List_test, iters=Iter, save_path=Result_save_Path, data_input_shape=data_input_shape, front_name='test_perfile')
            test_result = test_on_model4_subject(model=d_model, test_list=H5_List_test, iters=Iter, save_path=Result_save_Path, data_input_shape=data_input_shape, label_shape=label_shape, front_name='test', file_sep=file_sep,label_index=label_index)
            txt_s3.write(str(Iter) + '@' + str(test_result) + '\n')
            txt_s6.write(str(Iter) + '@' + str(test_result[0]) + '\n')
            txt_s7.write(str(Iter) + '@' + str(test_result[3]) + '\n')
            txt_s10.write(str(Iter) + '@' + str(test_result[4]) + '\n')

            # 换一个函数，因为test_on_model4_subject这个函数会保存预测结果，但是训练集没必要保存预测结果
            or_train_result = test_on_model4_subject4_or_train(model=d_model, test_list=H5_List_or_train, data_input_shape=data_input_shape, label_shape=label_shape,label_index=label_index)
            txt_s8.write(str(Iter) + '@' + str(or_train_result) + '\n')  # 保存or train 的 loss

            # verify =========================================================================
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

            # test =========================================================================
            if test_result[0] >= max_acc_test:
                max_acc_test_iter = Iter
                max_acc_test = test_result[0]
            if test_result[1] >= max_sen_test:
                max_sen_test_iter = Iter
                max_sen_test = test_result[1]
            if test_result[2] >= max_spc_test:
                max_spc_test_iter = Iter
                max_spc_test = test_result[2]
                # d_model.save(model_save_Path + '/m_' + str(Iter) + '_model.h5')
            if test_result[3] >= max_auc_test:
                max_auc_test_iter = Iter
                max_auc_test = test_result[3]
            if test_result[4] <= min_loss_test:
                min_loss_test_iter = Iter
                min_loss_test = test_result[4]
                # d_model.save(model_save_Path + '/m_' + str(Iter) + '_model.h5')



# 保存尽量新模型,防止训练中断
        if Iter % model_save_step == 0:
            d_model.save(model_save_Path + '/m_' + 'newest_model.h5')

# 每隔较长时间保存一次模型,用来做网络的可视化

        if Iter % model_save_step4_visualization == 0:
            d_model.save(model_save_Path + '/m_' + str(Iter) + '_model.h5')

# 优化策略:优化器变更以及学习率调整=========================================================================================
#       #开始的时候我们用的是adam,所以下面代码分为两部分,一部分是切换adam到sgd,另外一部分负责切换之后更新学习率
        #如果到达转换点,那么就开始转换,以防万一,先保存权重,之后重新编译模型,之后加载权重
        if Iter == optimizer_switch_point:
            d_model.save(model_save_Path + '/m_' + 'newest_model.h5')
            lr_new = lr_mod(Iter, max_epoch=50, epoch_file_size=trainset_num, batch_size=batch_size, init_lr=first_lr)
            d_model.compile(optimizer=SGD(lr=lr_new, momentum=0.9), loss=EuiLoss, metrics=[y_t, y_pre, Acc])
            d_model.load_weights(model_save_Path + '/m_' + 'newest_model.h5')

        if Iter > optimizer_switch_point:
            #batch_num_perepoch = or_train_num // batch_size  # 每个epoch包含的迭代次数,也即batch的个数
            lr_new = lr_mod(Iter, max_epoch=50, epoch_file_size=trainset_num, batch_size=batch_size, init_lr=first_lr)
            K.set_value(d_model.optimizer.lr, lr_new)



# 关闭文件,以供实时查看结果
        txt_s1.close()
        txt_s2.close()
        txt_s3.close()
        txt_s4.close()
        txt_s5.close()
        txt_s6.close()
        txt_s7.close()
        txt_s8.close()
        txt_s9.close()
        txt_s10.close()
        txt_s11.close()






