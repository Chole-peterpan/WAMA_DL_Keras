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







