clear
% 带有幅度衰减的局部加速下降抖动，的学习率加速下降调整;
max_epoch=40;%假设要跑完的总epoch数量
epoch_file_size = 3800;%每个epoch中训练集的文件数量
batch_size = 6;
init_lr = 0.0001;
doudong = 0.5;%控制抖动的系数,越小抖的越厉害，范围：大于0
min_lr_limitation = 1;%指定最终学习率降低到何种程度的参数，范围大于1，越大则学习率最后降低的越少
cos_ca = 0.5;%值在0到0.5之间，越接近0，则最终学习率越大，越接近0.5，则最终学习率越接近0



all_batch_num = floor(max_epoch*epoch_file_size/batch_size);%所有epoch包含的batch数
per_batch_num = floor(epoch_file_size / batch_size);%每个epoch包含的batch数
iters = 1:floor(all_batch_num);%batch的迭代次数
lr = zeros(1,floor(all_batch_num));
max_lr = (1+doudong)*init_lr;

for iter = 1:all_batch_num
    value = cos((rem(iter,per_batch_num)/(per_batch_num))*0.5*pi);
    %计算得到当前的epoch数
    current_epoch = iter/per_batch_num;
    %之后定下一个线性下降的趋势
    init_lr_tmp = init_lr*(1-current_epoch/max_epoch);
    %加入抖动：每个epoch都给一个cos生成的加速递减的系数，范围0~1;在加上一个常数项doudong，控制cos系数在总系数中的占比
    lr(iter)= (value+doudong)*init_lr_tmp;
    %加入基础值，为了不让学习率降为0
    lr(iter) = lr(iter)+ min_lr_limitation*max_lr;
    %归一化一下：
    lr(iter) = lr(iter)/((1+min_lr_limitation)*max_lr);
    %重新还原到正常学习率
    lr(iter) = lr(iter)*init_lr; 
    %以上只完成了抖动下降（虽然抖动也是局部的加速下降），下面进行加速下降的调整
    value1 = cos((iter/all_batch_num)*cos_ca*pi);
    lr(iter) = lr(iter)*value1;
end

figure;
plot(iters,lr);
axis([0 1.1*all_batch_num 0 1.2*max(lr)]);


% figure;
% value1 = cos(iters/(max_epoch*epoch_file_size)*cos_ca*pi);
% plot(iters,value1.*lr);
% axis([0 1.1*all_batch_num 0 1.2*max(lr)]);



% 学习率策略：
% 前2个epoch用ADAm
% 后面的用抖动加速衰减的SGD


