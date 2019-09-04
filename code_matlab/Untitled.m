% clear all
% 
% a = 4920591;
% b = -2000481;
% 
% pa = exp(a)/(exp(a)+exp(b));
% pb = exp(b)/(exp(a)+exp(b));
% s = rng;
% % unifrnd('seed',1)
% rng(s)
% % r = rand(1,5)
% unifrnd (1,2)

clear
workspace = load('H:\@data_NENs_recurrence\PNENs\data\@flow2\2block\15_2.mat');
data = workspace.block;
data = imresize3(data,  [size(data,1),size(data,2),size(data,3)],  'cubic');

augdict.savefomat.mode = 4;
augdict.savefomat.param = [500,500,60];

[aug_data,~,~] = aug43D(data,  augdict, 0,  []);



figure;
for i = 1:size(aug_data,3)

subplot(1,2,1);imshow(aug_data(:,:,i),[]);
% subplot(1,2,2);imshow(data(:,:,i),[]);
title(i);
   pause(0.1);
    
    
    
end




%% 
data1 = tumor_mat;
data2 = tumor_mat_other;

%断层查看一个立方体
figure;
for i = 1:size(data1,3)
%    imshow(data(:,:,i),[0,1]);
subplot(1,2,1);imshow(data1(:,:,i),[]);
subplot(1,2,2);imshow(data2(:,:,i),[]);
% imshowpair(data1)
   pause(0.2);
    
    
    
end

%%
or_path = 'H:\@data_NENs_recurrence\PNENs\data\4test\49_1.h5';
wokspace = load('H:\@data_NENs_recurrence\PNENs\data\2block\49_1');
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
subplot(2,2,1);imshow(data1(:,:,i),[0,1]);
subplot(2,2,2);imshow(data2(:,:,i),[0,1]);

subplot(2,2,3);imshow(data3(:,:,i),[0,1]);
subplot(2,2,4);imshow(data4(:,:,i),[0,1]);


% imshowpair(data1)
   pause(0.1);
   
end



