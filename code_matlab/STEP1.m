%% 整理NIFTI数据信息，并保存到mat文件中
%% 初始化
initialization;
data_path = {};
data_window = {};
%% 设置路径参数
% 模态or序列1
data_path{end+1} = 'G:\estdata\data\0nii';
data_window{end+1} = [-25,285];
% 模态or序列2
% data_path{end+1} = 'G:\estdata\data\0nii';
% data_window{end+1} = [-10,205];
% 模态or序列3
% data_path{end+1} = 'G:\estdata\data\0nii';
% data_window{end+1} = [-400,805];
% label文件
label_file = 'G:\estdata\data\label.xlsx';
% 保存路径
save_path = 'G:\estdata\1mat';
%% 设定其他参数
% 是否重采样体素size
resample_flag = true; 
% roi外扩的大小（mm）
extend_length = 0; 
% 空间体素重采样的目标size
adjust_voxelsize = [0.5,0.5,1.0]; 
%% 执行
STEP1_child;



