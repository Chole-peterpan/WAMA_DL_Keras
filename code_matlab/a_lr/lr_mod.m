function [lr] = lr_mod(iter,max_epoch,epoch_file_size,batch_size,init_lr ,doudong,min_lr_limitation,cos_ca)
% ���з���˥���ľֲ������½���������ѧϰ�ʼ����½�����;
% max_epoch=40;%����Ҫ�������epoch����
% epoch_file_size = 3800;%ÿ��epoch��ѵ�������ļ�����
% batch_size = 6;
% init_lr = 0.0001;
% doudong = 0.5;%���ƶ�����ϵ��,ԽС����Խ��������Χ������0
% min_lr_limitation = 1;%ָ������ѧϰ�ʽ��͵����̶ֳȵĲ�������Χ����1��Խ����ѧϰ����󽵵͵�Խ��
% cos_ca = 0.5;%ֵ��0��0.5֮�䣬Խ�ӽ�0��������ѧϰ��Խ��Խ�ӽ�0.5��������ѧϰ��Խ�ӽ�0



all_batch_num = floor(max_epoch*epoch_file_size/batch_size);%����epoch������batch��
per_batch_num = floor(epoch_file_size / batch_size);%ÿ��epoch������batch��
max_lr = (1+doudong)*init_lr;


value = cos((rem(iter,per_batch_num)/(per_batch_num))*0.5*pi);
%����õ���ǰ��epoch��
current_epoch = iter/per_batch_num;
%֮����һ�������½�������
init_lr_tmp = init_lr*(1-current_epoch/max_epoch);
%���붶����ÿ��epoch����һ��cos���ɵļ��ٵݼ���ϵ������Χ0~1;�ڼ���һ��������doudong������cosϵ������ϵ���е�ռ��
lr= (value+doudong)*init_lr_tmp;
%�������ֵ��Ϊ�˲���ѧϰ�ʽ�Ϊ0
lr = lr+ min_lr_limitation*max_lr;
%��һ��һ�£�
lr = lr/((1+min_lr_limitation)*max_lr);
%���»�ԭ������ѧϰ��
lr= lr*init_lr;
%����ֻ����˶����½�����Ȼ����Ҳ�Ǿֲ��ļ����½�����������м����½��ĵ���
value1 = cos((iter/all_batch_num)*cos_ca*pi);
lr = lr*value1;


end

