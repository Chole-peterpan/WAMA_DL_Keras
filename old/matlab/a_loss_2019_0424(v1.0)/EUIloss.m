function [loss] = EUIloss(y_pred_f,y_true_f,batchsize)
% lambd在实验中就是batchsize
%     y_true_t = y_true[:,0:1]
%     y_pred_t = y_pred[:,0:1]
%     y_true_f = K.flatten(y_true_t)
%     y_pred_f = K.flatten(y_pred_t)
% 
%     d = K.sum(K.sqrt(K.square(y_true_f - y_pred_f) + 1e-12))
%     a = K.cast(K.greater_equal(d, 0.5), dtype='float32')
%     b = K.cast(K.greater_equal(0.12, d), dtype='float32')
%     c = K.cast(K.greater_equal(0.3, d), dtype='float32')
%     #e = K.cast(y_pred_f, dtype='float32')
% 
%     loss = (2 + 4 * a - 0.5 * b - 1 * c) * d + 0.2 * y_pred_f *d
% 
% 
%     return loss
d = sqrt((y_true_f - y_pred_f)^2+1e-12);
a = double(d>=0.5);
b = double(d<0.12);
c = double(d<0.3);

loss = (2 + 4 * a - 0.5 * b - 1 * c) * d + 0.2 * y_pred_f *d;
loss = batchsize*loss;


end

