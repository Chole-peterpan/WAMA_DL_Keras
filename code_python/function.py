import os
import numpy as np
import random, h5py
from sklearn import metrics
import math
import prettytable as pt



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



# 学习率调整函数
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
                param_name == 'GPU_index' or
                param_name == 'aug_subject_path_othermode' or
                param_name == 'or_subject_path_othermode'):
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




# 从列表中获取batch的函数,专用于训练时H5文件的读取
def get_batch_from_list(batch_size, index_flag, filepath_list, epoch, label_index, data_input_shape, label_shape):
    # read data from h5
    # new_epoch_flag = 0
    filenamelist = []
    labeel = []
    data_input_c = np.zeros([batch_size] + data_input_shape + [1], dtype=np.float32)  # net input container
    label_input_c = np.zeros([batch_size] + label_shape)

    print('sub num:', len(filepath_list))

    for ii in range(batch_size):
        # read data from h5
        read_name = filepath_list[index_flag]
        # print(read_name)
        filenamelist.append(read_name)
        H5_file = h5py.File(read_name, 'r')
        batch_x = H5_file['data'][:]
        batch_y = H5_file[label_index][:]
        H5_file.close()
        labeel.append(batch_y)
        # put data into container
        batch_x_t = np.transpose(batch_x, (1, 2, 0))
        data_input_c[ii, :, :, :, 0] = batch_x_t[:, :, :]
        label_input_c[ii] = batch_y  # 有点微妙啊这个地方,有点二元数组的感觉
        # index plus 1 and check
        index_flag = index_flag + 1

        if index_flag == len(filepath_list):
            index_flag = 0
            epoch = epoch + 1
            random.shuffle(filepath_list)  # new



    return [epoch, index_flag, filepath_list, data_input_c, label_input_c, filenamelist, labeel]

# v2.2版本  测试函数,增加分batch跑的模式
# 分batch的具体操作,加入总样本10个,batchsize为3,则首先10//3 = 3,然后先跑3-1 = 2个batch
# 然后把最后一个batch和剩余一个余数为1的样本组合起来,变成一个batchsize = 原batchsize+余数
def test_on_model4_subject_new(model, test_list, iters, data_input_shape, label_shape,label_index,lossfunc,
                           batch_size  = 3,
                           id_savepath = None,
                           pre_savepath = None,
                           label_savepath = None,
                           loss_savepath = None,

                           per_block_name_savepath = None,
                           per_block_pre_savepath = None,
                           per_block_label_savepath = None,
                           per_block_loss_savepath = None,
                           or_train_flag = False):
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
    :param mode:batch的模式
    :return:

    保存预测值,同时保存最终指标
    精确到样本,指标计算单位为样本
    ps*写函数注释的时候,三个爽引号后直接回车就行,就会出现以上pycharm自动补充的函数说明
    """

    testtset_num = len(test_list)
    Num_list_test = list(range(testtset_num))


    # 以block为单位的,需要储存
    true_label = []
    pred_value = []
    loss_temp = []
    file_name = []


    indexxx = 0
    all_iters = testtset_num // batch_size  # 一共要迭代predict的次数
    res_num = testtset_num % batch_size  # 不能纳入完整batch的剩余样本数

    print(batch_size,testtset_num,res_num)
    # 首先把前all_iters - 1次predict了,等到最后一个batch的时候,把它和剩余的结合起来=====================================
    for iii in range(all_iters):
        # 如果是最后一个batch,那么就吧剩下的也加上
        if iii == all_iters-1:
            real_batch_size = batch_size+res_num
        else:
            real_batch_size = batch_size

        # 构建容器
        data_input_1 = np.zeros([real_batch_size] + data_input_shape + [1], dtype=np.float32)  # net input container
        label_input_1 = np.zeros([real_batch_size] + label_shape)

        for ii in range(real_batch_size):
            read_name = test_list[indexxx]

            # 储存文件名时候记得去除路径和扩展名
            tmpname = read_name.split('/')[-1]
            file_name.append(tmpname.split('.')[0])

            print(read_name)
            H5_file = h5py.File(read_name, 'r')
            batch_x = H5_file['data'][:]
            batch_y = H5_file[label_index][:]
            print(batch_y)
            H5_file.close()
            true_label.append(float(batch_y[0][0]))
            # put data into container
            batch_x_t = np.transpose(batch_x, (1, 2, 0))
            data_input_1[ii, :, :, :, 0] = batch_x_t[:, :, :]
            label_input_1[ii] = batch_y  # 有点微妙啊这个地方,有点二元数组的感觉
            # index plus 1 and check
            indexxx = indexxx + 1
        # 预测
        result_pre = model.predict_on_batch(data_input_1)
        result_loss = model.test_on_batch(data_input_1, label_input_1)
        loss_temp.append(float(result_loss[0]))
        for iiii in range(real_batch_size):
            pred_value.append(result_pre[iiii][0])

    # 重新得到每个block的label的预测值,注意,是onehot形式的,而且只是用于二分类,多分类需要修改代码
    true_label_onehot = np.zeros([testtset_num] + label_shape)
    pred_value_onehot = np.zeros([testtset_num] + label_shape)
    pred_value_onehot[:, 0]=(np.array(pred_value))
    pred_value_onehot[:, 1] = (1-np.array(pred_value))
    true_label_onehot[:, 0] = (np.array(true_label))
    true_label_onehot[:, 1] = (1-np.array(true_label))

    per_block_loss = lossfunc(tf.convert_to_tensor(true_label_onehot),tf.convert_to_tensor(pred_value_onehot))
    with tf.Session() as sess:
        loss_temp_perblock = (per_block_loss.eval())
    loss_temp_perblock = list(loss_temp_perblock) # 针对到每个block的loss






    patient_order = []
    patient_index = []
    # 算出样本数量和序号
    for read_num in Num_list_test:
        read_name = test_list[read_num]
        patient_order_temp = read_name.split('/')[-1]  # Windows则为\\
        patient_order_temp = patient_order_temp.split('_')[0][1:] # 清除前缀s
        # patient_order_temp = int(patient_order_temp)
        if patient_order_temp not in patient_order:
            patient_order.append(patient_order_temp)
            patient_index.append(int(patient_order_temp))

    # 整合patch到一病人为单位:
    # 根据样本序号分配并重新加入最终list，最后根据这个最终list来计算最终指标
    final_true_label = []
    final_pred_value = []
    final_loss = [] #精确到每个病人的loss

    patient_index.sort(reverse=False)
    for patient_id in patient_index:  # 首先遍历所有病人序号
        tmp_patient_prevalue = []
        tmp_patient_reallabel = []
        tmp_patient_loss = []
        for read_num in Num_list_test:
            # 在每个病人序号下,遍历所有patch的文件名对应的病例号,
            # 如果属于该病人,则append到临时的list,最终在对这个临时的list操作,得到属于这个病人的一个值
            read_name = test_list[read_num]
            tmp_index = read_name.split('/')[-1]
            tmp_index = tmp_index.split('_')[0][1:] # [1:]是为了清除前缀s
            tmp_index = int(tmp_index)
            if tmp_index == patient_id:
                tmp_patient_prevalue.append(pred_value[read_num])
                tmp_patient_reallabel.append(true_label[read_num])
                tmp_patient_loss.append(loss_temp_perblock[read_num])
                # 此时已经获得了对应第patient_id个样本的全部预测值
                # 暂时的策略为：计算预测均值，如果均值大于0.5则取最大值，反之取最小值 ； label任取一个加入
        final_true_label.append(tmp_patient_reallabel[0])
        final_loss.append(np.mean(tmp_patient_loss))
        mean_pre = np.mean(tmp_patient_prevalue)
        if mean_pre > 0.5:
            final_pred_value.append(np.max(tmp_patient_prevalue))
        elif mean_pre < 0.5:
            final_pred_value.append(np.min(tmp_patient_prevalue))
        elif mean_pre == 0.5:
            final_pred_value.append(0.5)

    # 根据最终list来计算最终指标
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    patient_num = patient_index.__len__()
    for nn in range(patient_num):
        t_label = final_true_label[nn]  # true label
        p_value = final_pred_value[nn]
        # ptnt_id = patient_index[nn]



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

    # =====================================================
    mean_loss_per_block = np.mean(loss_temp_perblock) #以block为单位的loss
    mean_loss = np.mean(final_loss) #以病人为单位的loss

    print('Sensitivity', Sensitivity)
    print('Specificity', Specificity)
    print('Accuracy', Accuracy)
    print('Loss', mean_loss)
    print('Loss(blocks)', mean_loss_per_block)


    if id_savepath is not None:
        txt_id = open(id_savepath, 'a')
        txt_id.write(str(iters) + '@' + str(patient_index) + '\n')
        txt_id.close()

    if label_savepath is not None:
        txt_lb = open(label_savepath, 'a')
        txt_lb.write(str(iters) + '@' + str(final_true_label) + '\n')
        txt_lb.close()

    if pre_savepath is not None:
        txt_pr = open(pre_savepath, 'a')
        txt_pr.write(str(iters) + '@' + str(final_pred_value) + '\n')
        txt_pr.close()

    if loss_savepath is not None:
        txt_pr = open(loss_savepath, 'a')
        txt_pr.write(str(iters) + '@' + str(final_loss) + '\n')
        txt_pr.close()

    # 还需要保存病人为单位的loss,所以需要新增一个文件名
    if per_block_name_savepath is not None:
        txt_pr = open(per_block_name_savepath, 'a')
        txt_pr.write(str(iters) + '@' + str(file_name) + '\n')
        txt_pr.close()

    if per_block_pre_savepath is not None:
        txt_pr = open(per_block_pre_savepath, 'a')
        txt_pr.write(str(iters) + '@' + str(pred_value) + '\n')
        txt_pr.close()

    if per_block_label_savepath is not None:
        txt_pr = open(per_block_label_savepath, 'a')
        txt_pr.write(str(iters) + '@' + str(true_label) + '\n')
        txt_pr.close()

    if per_block_loss_savepath is not None:
        txt_pr = open(per_block_loss_savepath, 'a')
        txt_pr.write(str(iters) + '@' + str(loss_temp_perblock) + '\n')
        txt_pr.close()



    if or_train_flag: # 如果是对ortrain进行测试,那么只返回以block为单位的loss就好
        return [Accuracy, Sensitivity, Specificity, Aucc, mean_loss_per_block]
    else:
        return [Accuracy, Sensitivity, Specificity, Aucc, mean_loss]



# 构建多GPU模型的函数
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



# 输入路径,以及指定的样本id的list,挑选对应样本的文件名以列表形式返回
def get_filelist_frompath(filepath, expname, sample_id = None):

    file_name = os.listdir(filepath)
    file_List = []

    if sample_id is not None:
        for file in file_name:
            if file.endswith('.'+expname):
                id = int(file.split('_')[0])
                if id in sample_id:
                    file_List.append(os.path.join(filepath, file))

    else:
        for file in file_name:
            if file.endswith('.'+expname):
                file_List.append(os.path.join(filepath, file))

    return file_List


# 输入路径,以及专属形式的txt文件路径,返回list,也可以和上一个函数一样,指定样本
def get_filelist_fromTXT(filepath, txt_path, sample_id = None):


    f = open(txt_path, "r")
    contents = f.readlines()
    f.close()

    file_List = []

    if sample_id is not None:
        for item in contents:
            content = item.strip()
            id = int(content.split('_')[0])
            if id in sample_id:
                file_List.append(os.path.join(filepath, content))

    else:
        for item in contents:
            content = item.strip()
            file_List.append(os.path.join(filepath, content))


    return file_List




import keras.backend as K
import matplotlib.pyplot as plt
file_sep = os.sep
# 运行后需要重新compile一次
def lr_finder(model, tmp_weight_path, filelist, batchsize, Epoch, label_shape,
              data_input_shape, label_index, iter = None, beta = 0.98, inc_mode = 'mult',
              lr_low=1e-6, lr_high=10,show_flag = False):


    model.save(tmp_weight_path + file_sep + 'tmp.h5')


    file_num = len(filelist)
    loss = []
    loss_smooth = []
    lr = []

    iternum = (Epoch*file_num)/batchsize
    if iter is not None:
        iternum = iter

    if inc_mode == 'mult':
        mult = (lr_high / lr_low) ** (1/iternum)
    elif inc_mode ==  'inc':
        inc = (lr_high - lr_low) * (1 / iternum)

    avg_loss = 0.
    index_flag = 0
    epoch = 0
    print('train_num is :',file_num,' all iters are:', int(iternum))
    K.set_value(model.optimizer.lr, lr_low)


    for i in range(int(iternum)+1):
        # 读取batch
        epoch, index_flag, filelist, data_input_c, \
        label_input_c, filenamelist, labeel = get_batch_from_list(batch_size = batchsize,
                                                                  index_flag = index_flag,
                                                                  filepath_list = filelist,
                                                                  epoch = epoch,
                                                                  label_index = label_index,
                                                                  data_input_shape = data_input_shape,
                                                                  label_shape = label_shape)

        # 训练
        cost = model.train_on_batch(data_input_c, label_input_c,class_weight={0:1,1:1})

        print('epoch:', epoch, r'/', Epoch, ',  iter',i + 1, r"/", int(iternum)+1, ',  loss:', cost[0])


        loss.append(cost[0])
        avg_loss = beta * avg_loss + (1 - beta) * cost[0]
        loss_smooth.append(avg_loss / (1 - beta ** i))

        now_lr = K.get_value(model.optimizer.lr)
        lr.append(now_lr)

        if inc_mode == 'mult':
            K.set_value(model.optimizer.lr, now_lr * mult)
        elif inc_mode == 'inc':
            K.set_value(model.optimizer.lr, now_lr + inc)



    # 还原权重
    model.load_weights(filepath=tmp_weight_path + file_sep + 'tmp.h5', by_name=True)


    if show_flag:
        plt.figure()
        loss = np.array(loss)
        loss_smooth = np.array(loss_smooth)
        lr = np.array(lr)
        plt.plot(lr, loss)
        plt.plot(lr, loss_smooth)
        plt.xscale('log')
        plt.show()

    return [loss,loss_smooth,lr]


# cos退火 + 循环(sgdr)
def lr_mod_cos(epoch_file_size, batchsize, lr_high = 1e-3, lr_low = 1e-7, warmup_epoch=5, loop_step=[1, 2, 4],
               max_contrl_epoch=65, show_flag = False, decay = 0.8):
    iternum = int((epoch_file_size * max_contrl_epoch) / batchsize) + 1

    pi = math.pi
    loop_step = np.array(loop_step)
    loop_num = len(loop_step)
    # 求出各个loop的iter数量
    warmup_iternum = int((epoch_file_size * warmup_epoch) / batchsize)  # 热身需要的迭代次数
    all_loop_iternum = iternum - warmup_iternum  # 循环退火的总迭代次数
    each_loop_iter = [int((i * all_loop_iternum) / np.sum(loop_step)) for i in loop_step]

    lr_new = []
    tmp_lr = (np.array(range(warmup_iternum)) / warmup_iternum) * (lr_high - lr_low) + lr_low
    lr_new = lr_new + list(tmp_lr)

    for i in range(loop_num):
        tmp_iter = (np.array(range(each_loop_iter[i])) / each_loop_iter[i])
        tmp_lr = ((((np.cos(tmp_iter * pi) + 1) / 2) * (lr_high*(decay**(i)) - lr_low)) + lr_low)
        lr_new = lr_new + list(tmp_lr)


    if show_flag:
        plt.figure()
        loss = np.array(lr_new)
        plt.plot(lr_new)
        plt.show()
    # 一次性返回所有lr的list

    return lr_new

def lr_mod_4sgdr(iter, lr_list):
    if iter >= len(lr_list):
        return np.array(lr_list).min()
    else:
        return lr_list[iter]




