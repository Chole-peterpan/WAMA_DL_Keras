import numpy as np
import math
import os
import keras.backend as K
import matplotlib.pyplot as plt
from ._function import get_batch_from_list

file_sep = os.sep
# 运行后需要重新compile一次
def lr_finder(model,
              data_loader,
              label_index,
              tmp_weight_path,

              filelist,
              batchsize,
              Epoch=1,
              iter = None,
              beta = 0.98,
              inc_mode = 'mult',
              lr_low=1e-6,
              lr_high=10,
              show_flag = False):

    model.save(tmp_weight_path + file_sep + 'tmp.h5')
    datageter = get_batch_from_list(batchsize, filelist, data_loader, label_index)

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
    epoch = 0
    print('train_num is :',file_num,' all iters are:', int(iternum))
    K.set_value(model.optimizer.lr, lr_low)


    for i in range(int(iternum)+1):
        # 读取batch
        data_input_c, label_input_c, filenamelist = next(datageter)
        # 训练
        cost = model.train_on_batch(data_input_c, label_input_c)

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













