% clear
% 带有幅度衰减的局部加速下降抖动，的学习率加速下降调整;
max_epoch=40;%假设要跑完的总epoch数量
epoch_file_size = 6200;%每个epoch中训练集的文件数量
batch_size = 6;
init_lr = 2e-6;%初始学习率
min_lr_limitation = 2.2;%函数运行的某一步中，指定最终学习率降低到何种程度的参数，范围大于0，越大则学习率最后降低的越少
%其实上面这个参数min_lr_limitation也是调节震动幅度的，越接近0震动幅度越大，属于粗调，也就是影响程度最大的调整
%min_lr_limitation这个参数其实还可以改变总体下降趋势的速度，如果是0~0.8的话，那么下降趋势为先加速再减速，如果大于0.8，那么就一直加速下降
doudong = 0.1;%控制抖动的系数,越小抖的越厉害，范围：大于0，这个参数属于细调，上面的min_lr_limitation属于粗调，即对震动幅度的改变很大
cos_ca = 0.3;%值在0到0.5之间，越接近0，则最终学习率越大，越接近0.5，则最终学习率越接近0



all_batch_num = floor(max_epoch*epoch_file_size/batch_size);%所有epoch包含的batch数
iters = 1:floor(all_batch_num);%batch的迭代次数
lr = zeros(1,floor(all_batch_num));


for iter = 1:60000
    lr(iter)=lr_mod(iter,max_epoch,epoch_file_size,batch_size,init_lr,doudong,min_lr_limitation,cos_ca);
    if iter >= all_batch_num
       lr(iter)= lr_mod(all_batch_num,max_epoch,epoch_file_size,batch_size,init_lr,doudong,min_lr_limitation,cos_ca);
    end
end

figure;
plot(lr,'m');



% figure;
% value1 = cos(iters/(max_epoch*epoch_file_size)*cos_ca*pi);
% plot(iters,value1.*lr);
axis([0 1.1*all_batch_num 0 1.2*max(lr)]);



% 学习率策略：
% 前2个epoch用ADAm
% 后面的用抖动加速衰减的SGD


