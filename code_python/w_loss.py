from keras import backend as K





def Acc(y_true, y_pred):
    y_pred_r = K.round(y_pred)
    return K.equal(y_pred_r, y_true)

def y_t(y_true, y_pred):
    return y_true

def y_pre(y_true, y_pred):
    return y_pred



#tensor_a=tf.convert_to_tensor(a)
# tensor:转为数组:
#with tf.Session() as sess:
#    print(pre_tensor.eval())

#用于二分类,输出为两个神经元
def EuiLoss(y_true, y_pred):
    y_true_t = y_true[:,0:1]
    y_pred_t = y_pred[:,0:1]
    y_true_f = K.flatten(y_true_t)
    y_pred_f = K.flatten(y_pred_t)

    d = K.sum(K.sqrt(K.square(y_true_f - y_pred_f) + 1e-12))
    a = K.cast(K.greater_equal(d, 0.5), dtype='float32')
    b = K.cast(K.greater_equal(0.12, d), dtype='float32')
    c = K.cast(K.greater_equal(0.3, d), dtype='float32')
    #e = K.cast(y_pred_f, dtype='float32')

    loss = (2 + 4 * a - 0.5 * b - 1 * c) * d + 0.2 * y_pred_f *d


    return loss


def EuclideanLoss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    loss = K.sum(K.square(y_true_f - y_pred_f))
    return loss



#label_tensor.shape()
#pre_tensor

#label_tensor_f = K.flatten(label_tensor)
#pre_tensor_f = K.flatten(pre_tensor)


#这个loss只能同于labelshape为1的数据,即网络输出为一个神经元
def EuiLoss_onelabel(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    d = K.sum(K.sqrt(K.square(y_true_f - y_pred_f) + 1e-12))
    a = K.cast(K.greater_equal(d, 0.5), dtype='float32')
    b = K.cast(K.greater_equal(0.12, d), dtype='float32')
    c = K.cast(K.greater_equal(0.3, d), dtype='float32')
    loss = (2 + 4 * a - 0.5 * b - 1 * c) * d + 0.2 * y_pred_f *d
    return loss







