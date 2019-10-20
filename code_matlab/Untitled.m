% clear all
% 
% a = 4920591;
% b = -2000481;
% s
% pa = exp(a)/(exp(a)+exp(b));
% pb = exp(b)/(exp(a)+exp(b));
% s = rng;
% % unifrnd('seed',1)
% rng(s)
% % r = rand(1,5)
% unifrnd (1,2)

% clear
% workspace = load('H:\@data_NENs_recurrence\PNENs\data\@flow2\2block\15_2.mat');
% data = workspace.block;
% data = imresize3(data,  [size(data,1),size(data,2),size(data,3)],  'cubic');
% 
% augdict.savefomat.mode = 4;
% augdict.savefomat.param = [500,500,60];
% 
% [aug_data,~,~] = aug43D(data,  augdict, 0,  []);
% 
% 
% 
% figure;
% for i = 1:size(aug_data,3)
% 
% subplot(1,2,1);imshow(aug_data(:,:,i),[]);
% % subplot(1,2,2);imshow(data(:,:,i),[]);
% title(i);
%    pause(0.1);
%     
%     
%     
% end
% 
% 
% 
% 
% %% 
% data1 = tumor_mat;
% data2 = tumor_mat_other;
% 
% %断层查看一个立方体
% figure;
% for i = 1:size(data1,3)
% %    imshow(data(:,:,i),[0,1]);
% subplot(1,2,1);imshow(data1(:,:,i),[]);
% subplot(1,2,2);imshow(data2(:,:,i),[]);
% % imshowpair(data1)
%    pause(0.2);
%     
%     
%     
% end

%%
close all
clear
or_path =       'H:\@data_NENs_recurrence\PNENs\data\flow1\3or_out\s9_v1_m1_b1.h5';
wokspace = load('H:\@data_NENs_recurrence\PNENs\data\flow1\2block\s58_v1_m1_b1.mat');
data3 = wokspace.block;
data4 = wokspace.block_othermode;

aug_data = h5read(or_path,'/data');
aug_data_othermode = h5read(or_path,'/data_othermode');
olabel_1=h5read(or_path,'/label_1');
olabel_2=h5read(or_path,'/label_2');
olabel_3=h5read(or_path,'/label_3');
olabel1=h5read(or_path,'/label1');
olabel2=h5read(or_path,'/label2');
olabel3=h5read(or_path,'/label3');
olabel=h5read(or_path,'/label');
data1 = aug_data;
data2 = aug_data_othermode;

figure;
for i = 1:size(data1,3)
%    imshow(data(:,:,i),[0,1]);
subplot(2,2,1);imshow(data1(:,:,i),[0,1]);title('h5_a');
subplot(2,2,2);imshow(data2(:,:,i),[0,1]);title('h5_v');

subplot(2,2,3);imshow(data3(:,:,i),[0,1]);title('mat_a');
subplot(2,2,4);imshow(data4(:,:,i),[0,1]);title('mat_v');


% imshowpair(data1)
   pause(0.2);
   
end


figure;
for i = 1:size(data3,3)
%    imshow(data(:,:,i),[0,1]);
% subplot(2,2,1);imshow(data1(:,:,i),[0,1]);
% subplot(2,2,2);imshow(data2(:,:,i),[0,1]);

subplot(2,1,1);imshow(data3(:,:,i),[0,1]);
subplot(2,1,2);imshow(data4(:,:,i),[0,1]);


% imshowpair(data1)
   pause(0.1);
   
end














data33 = imresize3(data3, size(data3).*6,  'cubic');
data33 = imresize3(data33, size(data3),  'cubic');
figure;
subplot(2,1,1);imshow(data3(:,:,11),[]);title('or');
subplot(2,1,2);imshow(data33(:,:,11),[]);title('after');



%% 
clear
wokspace = load('H:\@data_NENs_recurrence\PNENs\data\@@flow3\2block\49_1.mat');
data3 = wokspace.block;
data4 = wokspace.block_othermode;

augdict.savefomat.mode = 0;

% 随机剪裁
augdict.random_cut.flag = 1;
augdict.random_cut.p = 1;
augdict.random_cut.dim = [1,2,3];
augdict.random_cut.range = [0.5,0.5,0];


% 随机缩放
augdict.random_scale.flag = 1;
augdict.random_scale.p = 1;
augdict.random_scale.dim = [1,2,3];
augdict.random_scale.range_low = [1,1,1];
augdict.random_scale.range_high = [2,2,1];

%
[aug_data,aug_data_othermode,~] = aug43D(data3,  augdict,  1,  data4);



figure;
for i = 1:size(data3,3)
%    imshow(data(:,:,i),[0,1]);
subplot(2,2,1);imshow(data3(:,:,i),[0,1]);title('or');
subplot(2,2,2);imshow(data4(:,:,i),[0,1]);title('or');

subplot(2,2,3);imshow(aug_data(:,:,i),[0,1]);title('aug');
subplot(2,2,4);imshow(aug_data_othermode(:,:,i),[0,1]);title('aug');


% imshowpair(data1)
   pause(0.1);
   
end

a = 0
for i = 1 :length(subject)
    a = a+subject(i).all_aug_num;
end


data = [];
label = zeros(1,74);
label = label';
label(end-7:end)=1;

label=[];

data = [ID,label,predict];
data_sort = sortrows(data,1);
%% best acc
pre = predict;
label_1 = label;
tmp_pre=sort(pre);
div_value=(tmp_pre(1:end-1)+tmp_pre(2:end))/2;
final_acc=zeros(length(div_value),1);
% for ii=1:length(div_value)
%    div_pre_label=double(pre>div_value(ii));
%    right=div_pre_label==label_1;
%    final_acc(ii)=sum(right)/length(label_1);
% end
for ii=1:length(div_value)
   div_pre_label=double(pre>div_value(ii));
   [ sensitivity, specificity] = sen2spc( label_1 ,div_pre_label);
   final_acc(ii)=sum([sensitivity,specificity]);
end

index=div_value(final_acc==max(final_acc));
figure;
plot(div_value,final_acc);title(['max is :',num2str(max(final_acc)),',thre is:',num2str(index(1))]);
[X,Y,T,AUC] = perfcurve(label,data,1);
figure;
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification ')
legend(strcat('AUC:',num2str(AUC)),'Location','Best')

%%
clear
fileInfo = h5info('G:\estdata\3or\s3_v1_m2_b10.h5');

figure;
for i = 506:size(mask,3)
%    imshow(data(:,:,i),[0,1]);
imshow(mask(:,:,i),[0,1]);title(num2str(i));

   pause();
   
end


