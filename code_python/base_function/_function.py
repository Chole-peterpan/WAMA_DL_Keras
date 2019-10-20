import random
import numpy as np
import h5py
import os
from sklearn import metrics
import prettytable as pt


file_sep = os.sep


# global label_index


def label2onehot(label_index, class_num, label):
    """
    :param label_index: 使用地机组label
    :param class_num: 该label中的类别数
    :param label: 该样本的label数值,包含所有组的非onehot编码的label
    """
    onehot_label = list(np.zeros(int(class_num)))
    tmp_index = label[label_index - 1]
    onehot_label[int(tmp_index)] = 1
    return np.array(onehot_label)

def onehot2label(label):
    return np.argmax(label)

def load_data_from_h5(filename,label_index):
    H5_file = h5py.File(filename, 'r')
    data = H5_file['mode1'][:]
    label = H5_file['label'][:]
    class_num = H5_file['per_label_class_num'][label_index-1]
    H5_file.close()

    data = np.transpose(data, (1, 2, 0))
    label = label2onehot(label_index, class_num, label)
    data_input_shape = list(data.shape)+[1] # 最后一个[1]是通道数
    label_shape = [len(label)]

    return data, label, data_input_shape, label_shape, class_num

def get_batch_from_list(batch_size, filepath_list, data_loader, label_index):
    index_flag = 0

    # 先读一个数据,然后获得data_input_shape和label_shape
    tmp_data, tmp_label, data_input_shape, label_shape, c= data_loader(filepath_list[0],label_index)
    # 构建容器
    data_input_c = np.zeros([batch_size] + data_input_shape, dtype=np.float32)  # net input container
    label_input_c = np.zeros([batch_size] + label_shape)

    while True:
        filenamelist = []
        for ii in range(batch_size):
            # read data from h5
            read_name = filepath_list[index_flag]
            # print(read_name)
            filenamelist.append(read_name)
            batch_x ,batch_y ,a, b, c= data_loader(read_name,label_index)
            data_input_c[ii, :, :, :, 0] = batch_x # 注意,batchx是带通道的
            label_input_c[ii] = batch_y
            # index plus 1 and check
            index_flag = index_flag + 1
            if index_flag == len(filepath_list):
                index_flag = 0
                random.shuffle(filepath_list)  # new
        yield [data_input_c, label_input_c, filenamelist]

def keras_trainer(model, data, label):
    return model.train_on_batch(data, label)

def keras_predictor(model, data):
    return model.predict_on_batch(data)


def keras_tester4h5(model, batchsize, test_file_list, loss_func, data_loader, test_log_save_path,iter,label_index):

    # 先读一个数据,然后获得data_input_shape和label_shape
    tmp_data, tmp_label, data_input_shape, label_shape, c = data_loader(test_file_list[0],label_index)

    testtset_num = len(test_file_list)


    # 首先获得以block为单位的预测结果,需要储存
    file_name = []  # 每个block的文件名
    pred_value = [] # 每个block的预测值,onehot类型
    label_value = [] # 每个block的label,onehot类型
    loss_temp = [] # 每个batch的loss(注意不是block的,block的需要另外计算)

    true_label = [] # block的真实label
    pre_label = [] # block的预测label
    pre_label_pre = [] #预测class对应的pre
    true_label_pre = [] #真实class对应的pre

    indexxx = 0
    all_iters = testtset_num // batchsize  # 一共要迭代predict的次数
    res_num = testtset_num % batchsize  # 不能纳入完整batch的剩余样本数

    print(batchsize,testtset_num,res_num)
    # 首先把前all_iters - 1次predict了,等到最后一个batch的时候,把它和剩余的结合起来=====================================
    for iii in range(all_iters):
        # 如果是最后一个batch,那么就吧剩下的也加上
        if iii == all_iters-1:
            real_batch_size = batchsize+res_num
        else:
            real_batch_size = batchsize

        # 构建容器
        data_input_1 = np.zeros([real_batch_size] + data_input_shape, dtype=np.float32)  # net input container
        label_input_1 = np.zeros([real_batch_size] + label_shape)

        for ii in range(real_batch_size):
            read_name = test_file_list[indexxx]

            # 储存文件名时候记得去除路径和扩展名
            tmpname = read_name.split('/')[-1]
            file_name.append(tmpname.split('.')[0])

            print(read_name)
            batch_x, batch_y, a, b, c = data_loader(read_name,label_index)

            true_label.append(onehot2label(batch_y))
            # put data into container
            data_input_1[ii, :, :, :, 0] = batch_x
            label_input_1[ii] = batch_y  # 有点微妙啊这个地方,有点二元数组的感觉
            # index plus 1 and check
            indexxx = indexxx + 1
        # 预测
        result_pre = model.predict_on_batch(data_input_1)
        result_loss = model.test_on_batch(data_input_1, label_input_1)
        loss_temp.append(float(result_loss[0]))
        for iiii in range(real_batch_size):
            pred_value.append(list(result_pre[iiii]))
            label_value.append(list(label_input_1[iiii]))

    pre_label = list(np.argmax(np.array(pred_value),axis = 1))
    pre_label_pre = list(np.max(np.array(pred_value),axis = 1)) #预测class对应的pre
    for i in range(len(pre_label)):
        true_label_pre.append(pred_value[i][int(true_label[i])])


    # 计算每一个block的loss
    per_block_loss = loss_func(tf.convert_to_tensor(np.array(label_value,dtype=np.float64)),
                               tf.convert_to_tensor(np.array(pred_value,dtype=np.float64)))
    with tf.Session() as sess:
        loss_temp_perblock = (per_block_loss.eval())
    loss_temp_perblock = list(loss_temp_perblock) # 针对到每个block的loss

    # 接下来就是汇总到每一个病人==================================================
    patient_order = []
    patient_index = []
    # 提取样本序号
    for read_num in range(testtset_num):
        read_name = test_file_list[read_num]
        patient_order_temp = read_name.split('/')[-1]  # Windows则为\\
        patient_order_temp = patient_order_temp.split('_')[0][1:] # 清除前缀s
        if patient_order_temp not in patient_order:
            patient_order.append(patient_order_temp)
            patient_index.append(int(patient_order_temp))

    # 根据样本序号分配并重新加入最终list，最后根据这个最终list来计算最终指标
    final_true_label = [] #真正的label
    final_pre_label = []  #预测label
    final_pred_value = [] #预测的class对应的pre
    final_true_value = [] #真实class对应的pre
    final_loss = [] #精确到每个病人的loss
    final_pre = [] #精确到每个人的pre onehot形式

    patient_index.sort(reverse=False)
    # 首先遍历所有病人序号
    for patient_id in patient_index:

        tmp_true_label = []  # 该病人真正的label
        # tmp_pre_label = []  # 该病人预测label
        tmp_pred_value = []  # 该病人预测的label对应的pre
        tmp_true_value = []  # 该病人真实class对应的pre
        tmp_loss = []  # 该病人的loss

        # 便利所有文件,提取属于该病人的指标
        for read_num in range(testtset_num):
            # 在每个病人序号下,遍历所有patch的文件名对应的病例号,
            # 如果属于该病人,则append到临时的list,最终在对这个临时的list操作,得到属于这个病人的一个值
            read_name = test_file_list[read_num]
            tmp_index = read_name.split('/')[-1]
            tmp_index = tmp_index.split('_')[0][1:] # [1:]是为了清除前缀s
            tmp_index = int(tmp_index)
            if tmp_index == patient_id:
                tmp_true_label.append(true_label[read_num])#
                # tmp_pre_label.append(pre_label[read_num])#
                tmp_pred_value.append(pred_value[read_num])#
                # 注意预测label这个地方二分类和多分类是不同的,
                # 只能加入onehot的pre,不能直接加入预测的pre,
                # 因为预测label可能不是一个,所以加入对应不同class的pre,后续无法计算,
                # 所以要先把onehot的pre求均值,之后argmax得到汇总的label
                # ps:对于二分类,先判断均值,之后再去极值
                tmp_true_value.append(true_label_pre[read_num])#
                tmp_loss.append(loss_temp_perblock[read_num])

                # 此时已经获得了对应第patient_id个样本的全部预测值
                # 暂时的策略为：计算预测均值，如果均值大于0.5则取最大值，反之取最小值 ； label任取一个加入

        # 整合该病人所有block
        final_true_label.append(tmp_true_label[0])  # 真正的的label
        final_true_value.append(np.mean(tmp_true_value))  # 真实class对应的pre
        final_loss.append(np.mean(tmp_loss)) # 精确到每个病人的loss
        final_pre.append(list(np.mean(tmp_pred_value, axis=0)))
        # 预测pre球个均值,然后argmax
        if label_shape == 2: #此时是二分类,注意二分类预测label对应的预测值必然大于0.5
            # 首先判断属于0的pre的均值
            pfor0 = (np.mean(tmp_pred_value, axis=0))[0]
            if pfor0 > 0.5: # 如果p大于0.5,则证明属于0类,所以加入0类中的最大值
                final_pre_label.append(0)  # 预测label
                final_pred_value.append(np.max(np.array(tmp_pred_value)[:,0], axis=0)) # 预测的label对应的pre

            else : # 反之属于1类,append 1类中的最大值
                final_pre_label.append(1)  # 预测label
                final_pred_value.append(np.max(np.array(tmp_pred_value)[:, 1], axis=0))  # 预测的label对应的pre

        else: # 此时是多分类
            final_pre_label.append(np.argmax(np.mean(tmp_pred_value, axis=0)))  # 预测label,取均值之后argmax
            final_pred_value.append(np.max(np.mean(tmp_pred_value, axis=0)))  # 预测的label对应的pre,均值后max


    # 如果是二分类,则计算sen,spc,acc,auc,否则只计算acc
    # 以下计算指标都是以病人为单位的,默认概率是属于1类的概率,所以取第二列病人的预测值
    if label_shape[0] == 2:
        acc, sen, spc, auc = getAccSenSpcAuc(final_true_label, list(np.array(final_pre)[:, 1]))
    else:
        acc = 1 - (len(np.nonzero(np.array(final_true_label) - np.array(final_pre_label))[0]) / len(final_true_label))




    # 打印测试简略结果
    mean_loss_per_block = np.mean(loss_temp_perblock) #以block为单位的loss
    mean_loss = np.mean(final_loss) #以病人为单位的loss

    if label_shape[0] == 2:
        print('Sensitivity', sen)
        print('Specificity', spc)
        print('Accuracy', acc)
        print('Loss', mean_loss)
        print('Loss(blocks)', mean_loss_per_block)
    else:
        print('Accuracy', acc)
        print('Loss', mean_loss)
        print('Loss(blocks)', mean_loss_per_block)



    # 创建log文件名
    # 建立储存的logtxt
    # 4 block:===================================================================
    name_block = test_log_save_path + file_sep+'Block_name_.txt'  # # loss
    loss_block = test_log_save_path + file_sep+'Block_loss.txt'  # 文件名,
    t_label_block = test_log_save_path + file_sep+'Block_true_label.txt' # 真实label,
    p_label_block = test_log_save_path + file_sep+'Block_pre_label.txt' # 预测label,
    t_label_pre_block = test_log_save_path + file_sep+'Block_pre_4_true.txt'# 真实label对应的pre,
    p_label_pre_block = test_log_save_path + file_sep+'Block_pre_4_pre.txt'# 预测label对应的pre,
    onehot_pre_block = test_log_save_path + file_sep+'Block_pre_onehot.txt'# pre的onehot形式
    if label_shape[0] == 2: #如果是2分类,那么增加存储属于'1'类的pre
        pre_positive_block = test_log_save_path + file_sep+'Block_pre_4_positive.txt'

    # 4 person: =================================================================
    id_person = test_log_save_path + file_sep+'Person_id.txt'# id
    loss_person = test_log_save_path + file_sep + 'Person_loss.txt'# loss
    t_label_person = test_log_save_path + file_sep + 'Person_true_label.txt'# 真实label,
    p_label_person = test_log_save_path + file_sep + 'Person_pre_label.txt'# 预测label,
    t_label_pre_person = test_log_save_path + file_sep + 'Person_pre_4_true.txt'# 真实class对应的pre,
    p_label_pre_person = test_log_save_path + file_sep + 'Person_pre_4_pre.txt'# 预测label对应的pre,
    onehot_pre_person = test_log_save_path + file_sep + 'Person_pre_onehot.txt'# pre的onehot形式
    # 如果是2分类,那么增加存储属于'1'类的pre
    if label_shape[0] == 2:
        pre_positive_person = test_log_save_path + file_sep + 'Person_pre_4_positive.txt'  #


    # 写入log
    # 只需写入一次的文件
    if isempty4txt(name_block): # 如果是空的,则写入.否则不写入
        txt = open(name_block,'a') #block name
        txt.write(str(iter) + '@' + str(file_name) + '\n')
        txt.close()

        txt = open(t_label_block,'a') #block真实label
        txt.write(str(iter) + '@' + str(true_label) + '\n')
        txt.close()

        txt = open(id_person,'a') # person id
        txt.write(str(iter) + '@' + str(patient_index) + '\n')
        txt.close()

        txt = open(t_label_person,'a') # person真实label
        txt.write(str(iter) + '@' + str(final_true_label) + '\n')
        txt.close()

    # 需要不断写入的文件
    txt = open(loss_block, 'a')  # block的loss
    txt.write(str(iter) + '@' + str(loss_temp_perblock) + '\n')
    txt.close()

    txt = open(p_label_block, 'a')  # block的预测label
    txt.write(str(iter) + '@' + str(pre_label) + '\n')
    txt.close()

    txt = open(t_label_pre_block, 'a')  # 真实label对应的pre
    txt.write(str(iter) + '@' + str(true_label_pre) + '\n')
    txt.close()

    txt = open(p_label_pre_block, 'a')  # 预测label对应的pre
    txt.write(str(iter) + '@' + str(pre_label_pre) + '\n')
    txt.close()

    txt = open(onehot_pre_block, 'a')  # pre的onehot形式
    txt.write(str(iter) + '@' + str(pred_value) + '\n')
    txt.close()

    if label_shape[0] == 2:
        txt = open(pre_positive_block, 'a')  # block属于'1'类的pre
        txt.write(str(iter) + '@' + str(list(np.array(pred_value)[:, 1])) + '\n')
        txt.close()


    txt = open(loss_person, 'a')  # person的loss
    txt.write(str(iter) + '@' + str(final_loss) + '\n')
    txt.close()

    txt = open(p_label_person, 'a')  # # 预测label,
    txt.write(str(iter) + '@' + str(final_pre_label) + '\n')
    txt.close()

    txt = open(t_label_pre_person, 'a')  ## 真实class对应的pre,
    txt.write(str(iter) + '@' + str(final_true_value) + '\n')
    txt.close()

    txt = open(p_label_pre_person, 'a')  # 预测class对应的pre,
    txt.write(str(iter) + '@' + str(final_pred_value) + '\n')
    txt.close()

    txt = open(onehot_pre_person, 'a')  # 预测class对应的pre,
    txt.write(str(iter) + '@' + str(final_pre) + '\n')
    txt.close()

    if label_shape[0] == 2:
        txt = open(pre_positive_person, 'a')  # 预测class对应的pre,
        txt.write(str(iter) + '@' + str(list(np.array(final_pre)[:, 1])) + '\n')
        txt.close()

    if label_shape[0] == 2:
        return acc, sen, spc, auc, mean_loss
    else:
        return acc, 1  , 1  , 1  , mean_loss



# 改变字符串颜色的函数
def char_color(s,front,word):
    new_char = "\033[0;"+str(int(word))+";"+str(int(front))+"m"+s+"\033[0m"
    return new_char


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




def getAccSenSpcAuc(label, pre):
    final_true_label = label
    final_pred_value = pre
    # 根据最终list来计算最终指标
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    patient_num = len(label)
    for nn in range(patient_num):
        t_label = final_true_label[nn]  # true label
        p_value = final_pred_value[nn]

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

    return Accuracy,Sensitivity,Specificity,Aucc


def sigmoid_y(x):
    if x < 0.5:
        x = 0
    else:
        x = 1
    return x


# 如果为空,则返回0
def isempty4txt(filename):
    if not os.path.exists(filename):
        return True
    else:
        f = open(filename, "r")
        contents = f.readlines()
        f.close()
        if contents == []:
            return True
        else:
            return False



def color_printer(lr, taskname, epoch, Iter, per_epoch_batchnum, filenamelist, label_input_c, pre, cost):

    # 打印第一组列表
    tb = pt.PrettyTable()
    tb.field_names = [(char_color("task name", 50, 32)),
                      (char_color("lr", 50, 32)),
                      (char_color("epoch", 50, 32)),
                      (char_color("iter", 50, 32)),
                      (char_color("loss", 50, 32))]
    tb.add_row([taskname,
                lr,
                str(epoch)+'('+char_color(str(Iter%per_epoch_batchnum)+r'/'+str(per_epoch_batchnum),50,31)+')',
                Iter,
                cost[0]])

    tb.align["param_value"] = "l"
    tb.align["param_name"] = "r"
    print(tb)

    # 打印第二组列表
    tb = pt.PrettyTable()
    tb.field_names = [char_color('sub_subject', 50, 32), char_color('label', 50, 32), char_color('pre_value', 50, 32)]
    for ii in range(filenamelist.__len__()):
        sub_subject = filenamelist[ii].split('/')[-1]
        tb.add_row([sub_subject, label_input_c[ii], pre[ii]])
    print(tb)




