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
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss,EuiLoss_new
import keras.backend as K
from function import *
from sparsenet import SparseNetImageNet121
# step2: import extra model finished


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# from keras.utils.training_utils import multi_gpu_model   #导入keras多GPU函数


from keras.layers import Lambda, concatenate
from keras import Model
import tensorflow as tf

def multi_gpu_model(model, gpus):
  if isinstance(gpus, (list, tuple)):
    num_gpus = len(gpus)
    target_gpu_ids = gpus
  else:
    num_gpus = gpus
    target_gpu_ids = range(num_gpus)

  def get_slice(data, i, parts):
    shape = tf.shape(data)
    batch_size = shape[:1]
    input_shape = shape[1:]
    step = batch_size // parts
    if i == num_gpus - 1:
      size = batch_size - step * i
    else:
      size = step
    size = tf.concat([size, input_shape], axis=0)
    stride = tf.concat([step, input_shape * 0], axis=0)
    start = stride * i
    return tf.slice(data, start, size)

  all_outputs = []
  for i in range(len(model.outputs)):
    all_outputs.append([])

  # Place a copy of the model on each GPU,
  # each getting a slice of the inputs.
  for i, gpu_id in enumerate(target_gpu_ids):
    with tf.device('/gpu:%d' % gpu_id):
      with tf.name_scope('replica_%d' % gpu_id):
        inputs = []
        # Retrieve a slice of the input.
        for x in model.inputs:
          input_shape = tuple(x.get_shape().as_list())[1:]
          slice_i = Lambda(get_slice,
                           output_shape=input_shape,
                           arguments={'i': i,
                                      'parts': num_gpus})(x)
          inputs.append(slice_i)

        # Apply model on slice
        # (creating a model replica on the target device).
        outputs = model(inputs)
        if not isinstance(outputs, list):
          outputs = [outputs]

        # Save the outputs for merging back together later.
        for o in range(len(outputs)):
          all_outputs[o].append(outputs[o])

  # Merge outputs on CPU.
  with tf.device('/cpu:0'):
    merged = []
    for name, outputs in zip(model.output_names, all_outputs):
      merged.append(concatenate(outputs,
                                axis=0, name=name))
    return Model(model.inputs, merged)



# 读取外部数据集到list里面
test_path = r"/data/@data_pnens_recurrent_outside/a/4test"
file_name = os.listdir(test_path)
test_List = []
for file in file_name:
    if file.endswith('.h5'):
        test_List.append(os.path.join(test_path, file))


# 读取内部训练数据集到list里面
train_path = r"/data/@data_pnens_recurrent_new/data_test/a"
file_name = os.listdir(train_path)
train__List = []
for file in file_name:
    if file.endswith('.h5'):
        train__List.append(os.path.join(train_path, file))

H5_List_train = train__List
trainset_num = len(H5_List_train)
H5_List_test = test_List
# log的savepath
Result_save_Path = r'/data/XS_Aug_model_result/model_templete/recurrent/test_waibu/test_waibu10'


multi_gpu_mode = True #是否开启多GPU数据并行模式
gpu_munum = 3 #多GPU并行指定GPU数量

foldname = '1'  #暂时命名为1,不然观察不了
batch_size = 20
label_index = 'label3'
data_input_shape = [280,280,16]
label_shape = [2]
init_lr = 2e-5 # 初始学习率
trans_lr = 2e-5 # 转换后学习率
max_iter = 20000000 #最大迭代次数
print_steps = 50 # 打印训练信息的步长
task_name = 'test'  # 任务名称,自己随便定义
min_verify_Iters = 1  # 最小测试和验证迭代数量
verify_Iters_step = 50 # 测试和验证的步长

model_save_step = 300 # 模型保存的步长

optimizer_switch_point = 10000000000000 # 优化器转换的迭代数时间点
os_stage = "L"
if os_stage == "W":
    file_sep = r"\\"
elif os_stage == "L":
    file_sep = r'/'
else:
    file_sep = r'/'
log_save_Path = Result_save_Path + file_sep[0] + 'log'
model_save_Path = Result_save_Path + file_sep[0] + 'model'

os.system('mkdir '+log_save_Path)
os.system('mkdir '+model_save_Path)
# 开始训练:一下代码由main改变而来,但是不需要保存模型


if __name__ == "__main__":
    # prepare container:准备网络输入输出容器 暂时测试新getbatch函数,所以注释掉了
    # data_input_c = np.zeros([batch_size] + data_input_shape + [1], dtype=np.float32)  # net input container
    # label_input_c = np.zeros([batch_size] + label_shape)


    # 构建网络并compile
    # d_model_1 = vgg16_w_3d(use_bias_flag=True,classes=2)
    d_model = resnet_or(use_bias_flag=False,classes=2)
    # d_model_1 = resnext(classes=2, use_bias_flag=True)
    # d_model_1 = SparseNetImageNet121(classes = 2,activation='softmax',dropout_rate = 0.5)

    # d_model_1 = se_dense_net(nb_layers=[6, 12, 24, 16], growth_rate=32, nb_filter=64, bottleneck=True, reduction=0.0, dropout_rate=None, subsample_initial_block=True)


    if multi_gpu_mode:
        d_model = multi_gpu_model(d_model, gpus=gpu_munum)
    d_model.compile(optimizer=adam(lr=init_lr), loss='categorical_crossentropy', metrics=[y_t, y_pre, Acc])
    # pause()  # identify
    print(d_model.summary())  # view net
    # pause()  # identify
    # extra param initialization:初始化一些用来记录和显示的参数
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
    index_flag = 0  # 指向训练集列表的index,是更新epoch的依据
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

        # 读取batch ==================================================================================================================================
        epoch, index_flag, H5_List_train, data_input_c, label_input_c, filenamelist, labeel = get_batch_from_list(batch_size = batch_size,
                                                                                                                  index_flag = index_flag,
                                                                                                                  filepath_list = H5_List_train,
                                                                                                                  epoch = epoch,
                                                                                                                  label_index = label_index,
                                                                                                                  data_input_shape = data_input_shape,
                                                                                                                  label_shape = label_shape)
        # 读取完成 =====================================================================================================================================




        # train on model
        # losssss = d_model.test_on_batch(data_input_c,label_input_c)
        # print(Iter,'before train :',losssss)

        # 应该放在train前面,这样才和train返回的acc一样
        # 但是有个问题,家了dropout可能会影响train前向传播后pre与预测时pre不一致,进而导致acc也不一直
        pre = d_model.predict_on_batch(data_input_c)
        cost = d_model.train_on_batch(data_input_c, label_input_c)


        # losssss = d_model.test_on_batch(data_input_c,label_input_c)
        # print(Iter,'after train :',losssss)
        # losssss = d_model.test_on_batch(data_input_c,label_input_c)
        # print(Iter,'after train :',losssss)
        # losssss = d_model.test_on_batch(data_input_c,label_input_c)
        # print(Iter,'after train :',losssss)

        # print the detail of this iter===============================================================
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
            # # ver
            vre_result = test_on_model4_subject_new(model=d_model,
                                                test_list=H5_List_train,
                                                iters=Iter,
                                                data_input_shape=data_input_shape,
                                                label_shape=label_shape,
                                                id_savepath=ver_id_txt,
                                                label_savepath=ver_label_txt,
                                                pre_savepath=ver_pre_txt,
                                                label_index=label_index,
                                                batch_size=10)
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
                                                label_index=label_index,
                                                batch_size=10)
            # save
            txt_test_result.write(str(Iter) + '@' + str(test_result) + '\n')
            txt_test_loss.write(str(Iter) + '@' + str(test_result[4]) + '\n')
            # print(str(Iter) + '@' + str(test_result[4]))
            # print(test_result)


            # test or_train:这个函数不保存结果到文件,只返回loss
            # or_train_result = test_on_model4_subject4_or_train(model=d_model,
            #                                                    test_list=H5_List_or_train,
            #                                                    data_input_shape=data_input_shape,
            #                                                    label_shape=label_shape,
            #                                                    label_index=label_index)
            #
            # txt_or_loss.write(str(Iter) + '@' + str(or_train_result) + '\n')  # 保存or train 的 loss


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

        # 保存尽量新模型,防止训练中断
        if Iter % model_save_step == 0:
            # print(Iter)
            # 保存模型前输出一下当前batch的loss
            # losssss = d_model.test_on_batch(data_input_c,label_input_c)
            # print('before save loss from multigpu model: ',losssss)

            # 保存模型
            print('saving model')
            d_model.save(model_save_Path +file_sep +'m_' + 'newest_model.h5')
            print('already save')
            # 加载模型权重并重构multiGPU
            # d_model_1 = resnet_or(use_bias_flag=True, classes=2)
            # d_model_1.load_weights(model_save_Path + '/m_' + 'newest_model.h5', by_name=True)
            # d_model_1.compile(optimizer=adam(lr=init_lr), loss='categorical_crossentropy', metrics=[y_t, y_pre, Acc])
            # losssss = d_model_1.test_on_batch(data_input_c,label_input_c)
            # print('after save loss from singlegpu model: ',losssss)
            #


            # d_model = multi_gpu_model(d_model_1, gpus=3)
            # d_model.compile(optimizer=adam(lr=init_lr), loss='categorical_crossentropy', metrics=[y_t, y_pre, Acc])
            # d_model.load_weights(model_save_Path + '/m_' + 'newest_model.h5', by_name=True)
            # 用load后的权重输出一下loss,看模型保存前后loss是否相同
            # losssss = d_model.test_on_batch(data_input_c,label_input_c)
            # print('after load loss from multigpu model: ',losssss)



        # 每隔较长时间保存一次模型,用来做网络的可视化

        # if Iter % model_save_step4_visualization == 0:
        #     d_model.save(model_save_Path +file_sep+'m_' + str(Iter) + '_model.h5')

        # 优化策略:优化器变更以及学习率调整=========================================================================================
        # 开始的时候我们用的是adam,所以下面代码分为两部分,一部分是切换adam到sgd,另外一部分负责切换之后更新学习率
        # 如果到达转换点,那么就开始转换,以防万一,先保存权重,之后重新编译模型,之后加载权重
        if Iter == optimizer_switch_point:
            print('saving model')
            d_model.save(model_save_Path + '/m_' + 'newest_model.h5')
            print('saved succeed')

            lr_new = lr_mod(Iter, max_epoch=50, epoch_file_size=trainset_num, batch_size=batch_size, init_lr=trans_lr)
            # d_model_1 = resnet_or(use_bias_flag=True, classes=2)

            print('loading model weights')
            d_model.load_weights(model_save_Path + '/m_' + 'newest_model.h5',by_name=True)
            print('loaded succeed')

            # d_model = multi_gpu_model(d_model_1, gpus=3)
            d_model.compile(optimizer=SGD(lr=lr_new, momentum=0.9), loss='categorical_crossentropy', metrics=[y_t, y_pre, Acc])


        if Iter > optimizer_switch_point:
            #batch_num_perepoch = or_train_num // batch_size  # 每个epoch包含的迭代次数,也即batch的个数
            lr_new = lr_mod(Iter, max_epoch=50, epoch_file_size=trainset_num, batch_size=batch_size, init_lr=trans_lr)
            K.set_value(d_model.optimizer.lr, lr_new)



        # 关闭文件,以供实时查看结果
        txt_minibatch_loss.close()
        txt_ver_result.close()
        txt_test_result.close()
        txt_or_loss.close()
        txt_ver_loss.close()
        txt_test_loss.close()
        txt_lr.close()



