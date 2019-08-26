% clear
% ���з���˥���ľֲ������½���������ѧϰ�ʼ����½�����;
max_epoch=40;%����Ҫ�������epoch����
epoch_file_size = 6200;%ÿ��epoch��ѵ�������ļ�����
batch_size = 6;
init_lr = 2e-6;%��ʼѧϰ��
min_lr_limitation = 2.2;%�������е�ĳһ���У�ָ������ѧϰ�ʽ��͵����̶ֳȵĲ�������Χ����0��Խ����ѧϰ����󽵵͵�Խ��
%��ʵ�����������min_lr_limitationҲ�ǵ����𶯷��ȵģ�Խ�ӽ�0�𶯷���Խ�����ڴֵ���Ҳ����Ӱ��̶����ĵ���
%min_lr_limitation���������ʵ�����Ըı������½����Ƶ��ٶȣ������0~0.8�Ļ�����ô�½�����Ϊ�ȼ����ټ��٣��������0.8����ô��һֱ�����½�
doudong = 0.1;%���ƶ�����ϵ��,ԽС����Խ��������Χ������0�������������ϸ���������min_lr_limitation���ڴֵ��������𶯷��ȵĸı�ܴ�
cos_ca = 0.3;%ֵ��0��0.5֮�䣬Խ�ӽ�0��������ѧϰ��Խ��Խ�ӽ�0.5��������ѧϰ��Խ�ӽ�0



all_batch_num = floor(max_epoch*epoch_file_size/batch_size);%����epoch������batch��
iters = 1:floor(all_batch_num);%batch�ĵ�������
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



% ѧϰ�ʲ��ԣ�
% ǰ2��epoch��ADAm
% ������ö�������˥����SGD


