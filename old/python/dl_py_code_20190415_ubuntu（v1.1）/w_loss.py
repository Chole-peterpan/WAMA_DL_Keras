from keras import backend as K


def EuiLoss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    d = K.sum(K.sqrt(K.square(y_true_f - y_pred_f) + 1e-12))
    a = K.cast(K.greater_equal(d, 0.5), dtype='float32')
    b = K.cast(K.greater_equal(0.12, d), dtype='float32')
    c = K.cast(K.greater_equal(0.3, d), dtype='float32')
    loss = (2 + 4 * a - 0.5 * b - 1 * c) * d + 0.2 * y_pred_f *d
    return loss











