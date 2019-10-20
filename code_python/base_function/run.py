import keras.backend as K
import os
from ._function import get_batch_from_list

def Acc(y_true, y_pred):
    y_pred_r = K.round(y_pred)
    return K.equal(y_pred_r, y_true)
file_sep = os.sep

def run(model, label_index, trainer, tester, predictor,data_loader,taskname,
        loss_func, optimizer,
        train_file_list,
        test_file_list = None,
        ver_file_list = None,
        ortrain_file_list = None,
        log_save_path = None,
        train_batchsize = 1,
        test_batchsize = 1,
        train_epoch = 10,
        epoch_size = None,
        printer = None,
        lr_plan = None,

        min_test_epoch = 1,
        min_test_iter = None,

        test_epoch_step = 1,
        test_iter_step = None,

        vis_model_save_epoch_start = 1,
        vis_model_save_iter_start = None,
        vis_model_save_epoch_step = 1,
        vis_model_save_iter_step = None,

        newest_model_save_epoch_start = 1,
        newest_model_save_iter_start = None,
        newest_model_save_epoch_step = 1,
        newest_model_save_iter_step = None):
    """
    训练&验证&测试 函数
    :param model: 无需经过compile的模型 
    :param loss_func: 损失函数
    :param data_loader: 文件读取函数,要求输入为文件名,输出为[data, label, 数据shape],数据shape为形如(w,h,c)
    :param train_file_list: 训练集的文件名list(要求list中文件名是完整路径) 
    :param test_file_list: 测试集的文件名list(要求list中文件名是完整路径) ,若为None则不测试 
    :param ver_file_list: 验证集的文件名list(要求list中文件名是完整路径)  ,若为None则不验证
    :param ortrain_file_list: 未扩增的训练集的文件名list(要求list中文件名是完整路径) ,若为None则不测试
    :param log_save_path: log存放的文件夹
    :param train_batchsize 
    :param test_batchsize 
    :param GPU_index: 形如 '0',或 '0,1,2'
    :param use_muilti_GPU: 是否使用多GPU数据并行模式(并行可增大batchsize容量上限)
    :param train_epoch: 训练的总epoch数
    :param epoch_size: epochsize, 默认使用ortrain的size,如果ortrain为none,则使用train的size
    :param printer: 打印函数,要求形如:
                    def printer(epoch, iter, train_result, ver_result, test_result):
    :param lr_plan: 学习率计划函数, 提供模型接口,函数形如:
                    def lr_plan(epoch, iter):
                        return lr
                    
    :param min_test_epoch: 最小测试&验证epoch,最终要转化为迭代数
    :param min_test_iter: 如果规定迭代数,那么就只看迭代数
    :param test_epoch_step: 
    :param test_iter_step: 
    :param vis_model_save_epoch_start: 
    :param vis_model_save_iter_start: 
    :param vis_model_save_epoch_step: 
    :param vis_model_save_iter_step: 
    :param newest_model_save_epoch_start: 
    :param newest_model_save_iter_start: 
    :param newest_model_save_epoch_step: 
    :param newest_model_save_iter_step
    :return: 
    """

    # 构建一堆log储存文件
    log_save_Path = log_save_path + file_sep + 'log'
    model_save_Path = log_save_path + file_sep + '@model'
    test_log_save_path = log_save_Path + file_sep + '@test'
    ver_log_save_path = log_save_Path + file_sep + '@ver'
    or_log_save_path = log_save_Path + file_sep + '@or'
    os.system('mkdir ' + log_save_Path)
    os.system('mkdir ' + model_save_Path)
    os.system('mkdir ' + test_log_save_path)
    os.system('mkdir ' + ver_log_save_path)
    os.system('mkdir ' + or_log_save_path)

    lr_txt = log_save_Path + file_sep + '@' + '1' + '_lr.txt'  # 学习率曲线  txt_s11
    minibatch_loss_txt = log_save_Path + file_sep+'@' + '1' + '_loss.txt'  # minibatch上的loss  txt_minibatch_loss
    or_loss_txt = log_save_Path + file_sep+'@' + '1' + '_loss_or.txt'  # 在原未扩增训练集上的loss  txt_or_loss
    or_result_txt = log_save_Path + file_sep+'@' + '1' + '_result_or.txt'  # 原未扩增训练集的所有指标  txt_ver_result
    ver_loss_txt = log_save_Path + file_sep+'@' + '1' + '_loss_ver.txt'  # 在验证集上的loss  txt_ver_loss
    ver_result_txt = log_save_Path + file_sep+'@' + '1' + '_result_ver.txt'  # 验证集的所有指标  txt_ver_result
    test_loss_txt = log_save_Path + file_sep+'@' + '1' + '_loss_test.txt'  # 在测试集上的loss  txt_test_loss
    test_result_txt = log_save_Path + file_sep + '@' + '1' + '_result_test.txt'  # 测试集的所有指标  txt_test_result


    # 先得到epoch size
    if epoch_size != None:
        print('epoch size is :', epoch_size)
    else:
        if ortrain_file_list == None:
            epoch_size = len(train_file_list)
            print('epoch size is :',epoch_size)
        else:
            epoch_size = len(ortrain_file_list)
            print('epoch size is :', epoch_size)

    # 计算每个epoch包含多少个batch(也就是一个epoch要跑多少次迭代数)
    per_epoch_batchnum = int(epoch_size / train_batchsize)+1
    if per_epoch_batchnum ==0:
        per_epoch_batchnum = 1
    # 再计算得到需要迭代的总次数
    all_train_iter = int((epoch_size * train_epoch) / train_batchsize)
    # 得到最小验证迭代数
    if min_test_iter is None:
        min_test_iter = min_test_epoch*per_epoch_batchnum
    # 计算验证迭代数步长
    if test_iter_step is None:
        test_iter_step = test_epoch_step*per_epoch_batchnum
    # 计算vis model保存最小迭代数
    if vis_model_save_iter_start is None:
        vis_model_save_iter_start = vis_model_save_epoch_start*per_epoch_batchnum
    # 计算vis model保存步长
    if vis_model_save_iter_step is None:
        vis_model_save_iter_step = vis_model_save_epoch_step*per_epoch_batchnum
    # 计算newest model保存最迭代数
    if newest_model_save_iter_start is None:
        newest_model_save_iter_start = newest_model_save_epoch_start*per_epoch_batchnum
    # 计算newest model保存步长
    if newest_model_save_iter_step is None:
        newest_model_save_iter_step = newest_model_save_epoch_step*per_epoch_batchnum


    # 构建训练集用的batchloader:使用yiled
    train_batchloader = get_batch_from_list(batch_size = train_batchsize,
                                            filepath_list = train_file_list,
                                            data_loader = data_loader,
                                            label_index = label_index)


    # 开始迭代
    for iter in range(all_train_iter):
        # 计算得到当前所属epoch
        epoch  = (iter//per_epoch_batchnum)

        # 如果有lr_plan,则设置学习率
        if lr_plan is not None:
            K.set_value(model.optimizer.lr, lr_plan(iter))

        # 获取batch
        data_input_c, label_input_c ,filenamelist = next(train_batchloader)

        # 输出当前batch的预测值
        minibatch_pre = predictor(model, data_input_c)#model.predict_on_batch(data_input_c)

        # 训练
        minibatch_cost =trainer(model, data_input_c, label_input_c) # model.train_on_batch(data_input_c, label_input_c)

        # 验证
        if iter >= min_test_iter and iter % test_iter_step == 0:
            if test_file_list is not None:
                test_result  = tester(model, test_batchsize, test_file_list, loss_func, data_loader,
                                      test_log_save_path, iter,label_index)
            else:
                test_result = None

        # 测试
        if iter >= min_test_iter and iter % test_iter_step == 0:
            if ver_file_list is not None:
                ver_result = tester(model, test_batchsize, ver_file_list, loss_func, data_loader,
                                    ver_log_save_path, iter,label_index)
            else:
                ver_result = None

        # 测试(ortrain)
        if iter >= min_test_iter and iter % test_iter_step == 0:
            if ortrain_file_list is not None:
                or_result = tester(model, test_batchsize, ortrain_file_list, loss_func, data_loader,
                                   or_log_save_path, iter,label_index)
            else:
                or_result = None

        # 如果有visstep,则每过一次保存
        if iter >= vis_model_save_iter_start and iter % vis_model_save_iter_step == 0:
            model.save(model_save_Path + file_sep + 'm_' + str(iter) + '_model.h5')

        # 如果有visstep,则每过一次保存
        if iter >= newest_model_save_iter_start and iter % newest_model_save_iter_step == 0:
            model.save(model_save_Path + file_sep + 'm_newest_model.h5')


        # 打印
        printer(K.get_value(model.optimizer.lr),
                taskname, epoch, iter, per_epoch_batchnum,
                filenamelist, label_input_c, minibatch_pre, minibatch_cost)



        # 记录到log
        txt_minibatch_loss = open(minibatch_loss_txt, 'a')
        txt_lr = open(lr_txt, 'a')
        txt_or_loss = open(or_loss_txt, 'a')
        txt_ver_loss = open(ver_loss_txt, 'a')
        txt_test_loss = open(test_loss_txt, 'a')
        txt_or_result = open(or_result_txt, 'a')
        txt_ver_result = open(ver_result_txt, 'a')
        txt_test_result = open(test_result_txt, 'a')

        txt_minibatch_loss.write(str(minibatch_cost[0]) + '\n')
        txt_lr.write(str(K.get_value(model.optimizer.lr)) + '\n')
        if iter >= min_test_iter and iter % test_iter_step == 0:
            if ver_file_list is not None:
                txt_ver_result.write(str(iter) + '@' + str(ver_result) + '\n')
                txt_ver_loss.write(str(iter) + '@' + str(ver_result[4]) + '\n') #注意,如果是多分类,那么也要在第4位返回loss
            if test_file_list is not None:
                txt_test_result.write(str(iter) + '@' + str(test_result) + '\n')
                txt_test_loss.write(str(iter) + '@' + str(test_result[4]) + '\n')
            if ortrain_file_list is not None:
                txt_or_result.write(str(iter) + '@' + str(or_result) + '\n')
                txt_or_loss.write(str(iter) + '@' + str(or_result[4]) + '\n')

        txt_minibatch_loss.close()
        txt_lr.close()
        txt_or_loss.close()
        txt_ver_loss.close()
        txt_test_loss.close()
        txt_or_result.close()
        txt_ver_result.close()
        txt_test_result.close()
























