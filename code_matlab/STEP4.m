%% 随机扩增，保证扩增后各类中数量平衡，各类中个样本数量平衡
%% 初始化
initialization;

%% 设置路径
block_mat_path =  'G:\estdata\2block';
augdict.mat_savepath =    'G:\estdata\4aug';

%% 设置其他扩增参数

%设定数据输出的模式以及size
augdict.savefomat.mode = 3;
augdict.savefomat.param = [150,150,80];

%总扩增数量
augdict.aug_num = 1000;
%以第几个label的分类来进行平衡
augdict.balance_label_index = 2;
%平衡时候各类的比例（label值从小到大）
augdict.a_b_ratio = [1,1];%最终扩增数量a比b，例如AB比例为1:2，则设置为[1,2],A是非PD，B是PD

% 扩增：旋转
augdict.rotation.flag = true;%是否使用旋转扩增
augdict.rotation.range = [0,180];%旋转扩增的角度范围
augdict.rotation.p = 1.0;

% 扩增：对比度调整
augdict.gray_adjust.flag = true;%是否使用对比度调整
augdict.gray_adjust.up = [0.9,1];%对比度调整的上界范围
augdict.gray_adjust.low = [.0,0.1];%对比度调整的下界范围
augdict.gray_adjust.p = 1.0;

% 扩增：左右反转 LEFT RIGHT
augdict.LR_overturn.flag = true;%是否使用左右翻转
augdict.LR_overturn.p = 1.0;%左右翻转的概率

% 扩增：上下翻转 UP DOWN
augdict.UD_overturn.flag = true;%是否使用上下翻转
augdict.UD_overturn.p = 1.0;%上下翻转的概率

% 扩增：随机剪切
augdict.random_cut.flag = true;%是否进行剪裁
augdict.random_cut.p = 1.0;
augdict.random_cut.dim = [1,2,3]; 
augdict.random_cut.range = [0.1,0.2,0.3];

% 扩增：随机拉伸
augdict.random_scale.flag = true;%是否进行剪裁
augdict.random_scale.p = 1.0;
augdict.random_scale.dim = [1,2,3];
augdict.random_scale.range_low = [0.5, 0.6, 0.7];
augdict.random_scale.range_high = [1.1, 1.2, 1.3];

%% run
STEP4_child;


