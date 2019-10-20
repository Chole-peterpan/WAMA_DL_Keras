%% 生成未扩增的数据
%% 初始化
initialization;
%% 设置路径
block_mat_path =  'G:\estdata\2block';
h5_savepath =    'G:\estdata\3or';

%% 设置其他参数
%设定数据输出的模式以及size
augdict.savefomat.mode = 3;
augdict.savefomat.param = [150,150,80];

%% run
STEP3_child;




