clear;
clc;
mode1_mat = 'G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@data\v\3cut\305.mat';
mode2_mat = 'G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@data\a\3cut\5.mat';
worksp1 = load(mode1_mat);
worksp2 = load(mode2_mat);
data1 = worksp1.Data_cu;
data2 = worksp2.Data_cu;

figure;
for iii= 1:size(data1,3)
subplot(3,1,1);
imshowpair(data1(:,:,iii), data2(:,:,iii), 'falsecolor');
subplot(3,1,2);
imshow(data1(:,:,iii),[]);
subplot(3,1,3);
imshow(data2(:,:,iii),[]);
pause();
end













