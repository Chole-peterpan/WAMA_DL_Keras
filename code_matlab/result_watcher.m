%% 观察最终结果的watcher，和realtimewatcher不一样，这个是针对样本观察的
% 分为3各级别：1）全体&针对到各个病人，2）针对到单个病人的各个肿瘤，3）针对到某个肿瘤各个block

%% 初始化
close all;
clear;
clc;

%% 指定参数
logpath = 'H:\qwe\log';
foldname = 1;
test_flag = true;
acc_thresold = 0.5;% 准确率阈值

%% 加载数据
result_watcher_1

%% 第一级别显示：以人为单位的loss，pre，acc（以块做平均）
% 设定参数
scale_size = 300;%loss热力图缩放的比例
% 显示loss以及acc
result_watcher_2;



%% 第二级别显示：进一步设定参数
% 显示病人id
uniq_id = unique(block_name(:,1))';
disp(['id:',num2str(uniq_id)]);
disp(person_id);


scale_size = 300;%loss热力图缩放的比例
% 设定参数
s_id = 51;%指定病人序号，非正数或0则不显示
v_id = 1;%指定扫描段序号，非正数或0则不显示
m_id = 1;%指定肿瘤序号，非正数或0则不显示
%若为正数n，则显示id为n的病人的所有块的情况（acc，以及loss），显示loss有两种方法，一种是多条曲线，一种是heatmap，heatmap需要自己变换为图片形式显示
%显示的时候注意排序，即同一肿瘤的不同块最好按照原来的空间相对位置排序显示，这样热力图的纵轴就和肿瘤切片的排序一致

show_flag = true;%如果为true，则显示各种指标
result_watcher_3;






















