def test_on_model(model, test_list,iters, save_path, data_input_shape,front_name):
    data_input_1 = np.zeros([1] + data_input_shape + [1], dtype=np.float32)  # net input container

    pred_txt = save_path + '/' +front_name +'predict_' + str(iters) + '.txt'
    orgi_txt = save_path + '/' +front_name +'orginal_' + str(iters) + '.txt'
    file_txt = save_path + '/' +front_name +'filename_' + str(iters) + '.txt'
    value_txt = save_path + '/' + front_name +'result_' + str(iters) + '.txt'

    txt_s1 = open(pred_txt, 'w')
    txt_s2 = open(orgi_txt, 'w')
    txt_s3 = open(file_txt, 'w')
    txt_s4 = open(value_txt, 'w')

    testtset_num = len(test_list)
    Num_list_test = list(range(testtset_num))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    true_label = []
    pred_value = []

    # wds-20190121
    patient_order = []
    patient_pre = []    #平均预测值，和block个数


    for read_num in Num_list_test:
        read_name = test_list[read_num]
        H5_file = h5py.File(read_name, 'r')
        batch_x = H5_file['data'][:]
        batch_y = H5_file['label_3'][:]
        print(batch_y)
        batch_x_t = np.transpose(batch_x, (1, 2, 0))
        data_input_1[0, :, :, :, 0] = batch_x_t[:, :, 0:16]
        H5_file.close()

        result_pre = model.predict_on_batch(data_input_1)

        true_label.append(float(batch_y[:]))
        pred_value.append(float(result_pre[:]))
        txt_s1.write(str(float(result_pre))+'\n')
        txt_s2.write(str(float(batch_y)) + '\n')
        txt_s3.write(read_name + '\n')

        y = batch_y
        result_pre = sigmoid_y(result_pre)
        if (y == 1) and (y == result_pre):
            tp = tp + 1  # 真阳
        elif (y == 0) and (y == result_pre):
            tn = tn + 1  # 真阴
        elif (y == 1) and (result_pre == 0):
            fn = fn + 1  # 假阴
        elif (y == 0) and (result_pre == 1):
            fp = fp + 1  # 假阳

        print('Sample_name', read_name)
        # print('Sample_label', y)
        # print('Sample_pre_label', result_pre)
        # print('num', read_num, result)
        # print(d_model.predict_on_batch(data_input_1))
        # accuracy, sensitivity, specificity

        #wds-20190121
        patient_order_temp = read_name.split('/')[-1]#Windows则为\\
        patient_order_temp = patient_order_temp.split('_')[0]
        if patient_order_temp not in  patient_order:
            patient_order.append(patient_order_temp)
            patient_pre.append([result_pre,1,y])
        else:
            index_temp = patient_order.index(patient_order_temp)
            patient_pre[index_temp][0] = \
                (patient_pre[index_temp][0]*patient_pre[index_temp][1]+result_pre)/(patient_pre[index_temp][1]+1)
            patient_pre[index_temp][1] = patient_pre[index_temp][1]+1

    Sensitivity = tp / ((tp + fn)+(1e-16))
    Specificity = tn / ((tn + fp)+(1e-16))
    Accuracy = (tp + tn) / ((tp + tn + fp + fn)+(1e-16))
    if (sum(true_label)==testtset_num) or (sum(true_label)==0):
        Aucc = 0
        print('only one class')

    else:
        Aucc = metrics.roc_auc_score(true_label, pred_value)
        print('AUC', Aucc)

    print('Sensitivity', Sensitivity)
    print('Specificity', Specificity)
    print('Accuracy', Accuracy)

    txt_s4.write('acc:'+str(Accuracy) + '\n')
    txt_s4.write('spc:' + str(Specificity) + '\n')
    txt_s4.write('sen:' + str(Sensitivity) + '\n')
    txt_s4.write('auc:' + str(Aucc) + '\n')

    # wds-20190121
    for i in range(len(patient_order)):
        #print('Patient:',patient_order[i],'Pre:',patient_pre[i][0],'label:',patient_pre[i][2],'Block_num:',patient_pre[i][1])
        print("Patient:%2s   Pre:%5.3f   Label:%d   Block_num:%d" %(patient_order[i],patient_pre[i][0], patient_pre[i][2],patient_pre[i][1]))


    return [Accuracy, Sensitivity, Specificity, Aucc]