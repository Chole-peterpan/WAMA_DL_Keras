%% 类似于实时扩增，所以就不需要数量计算的那么麻烦，直接指定扩增方法和扩增方法对应的参数就好
% 这样虽然不像step3那样严格保证分布，但是却方便了许多
%% 初始化
clc;
clear;

%% 参数设置
block_mat_path = 'H:\@data_NENs_recurrence\PNENs\data\2block';%block
augdict.mat_savepath = 'H:\@data_NENs_recurrence\PNENs\data\3aug20000';

augdict.aug_num = 500;%总扩增数量
augdict.class_a_id = 1:49;% 手动传入a类病人的id
augdict.class_b_id = 50:59;% 手动传入b类病人的id
augdict.a_b_ratio = [1,1];%a比b，例如AB比例为1:2，则设置为[1,2],A是非PD，B是PD

% 扩增：旋转
augdict.rotation.flag = true;%是否使用旋转扩增
augdict.rotation.range = [0,40];%旋转扩增的角度范围
augdict.rotation.p = 0.8;

% 扩增：对比度调整
augdict.gray_adjust.flag = true;%是否使用对比度调整
augdict.gray_adjust.up = [0.8   ,   1];%对比度调整的上界范围
augdict.gray_adjust.low = [0    ,   0.1];%对比度调整的下界范围
augdict.gray_adjust.p = 0.5;

% 扩增：左右反转 LEFT RIGHT
% 至于概率，随机生成均匀分布的0到1，如果是百分之5概率翻转，那就出现小于0.05的数字就翻转
augdict.LR_overturn.flag = true;%是否使用左右翻转
augdict.LR_overturn.p = 0.5;%左右翻转的概率


% 扩增：上下翻转 UP DOWN
augdict.UD_overturn.flag = true;%是否使用上下翻转
augdict.UD_overturn.p = 0.5;%上下翻转的概率

% 扩增最后返回的形式限制，输入是各种大小，要求返回的shape相同
% 0：不改变形状
% mode 1 是 3Dresize
% mode 2 是 容器居中
augdict.savefomat.mode = 2;
augdict.savefomat.param = [280,280,16];


% 随机数种子，以便复现实验
augdict.s = rng;



%% 预建立结构体数组，储存block名称以及该block的扩增参数，这样就能精确定位每一个block的扩增参数了
block_detail = [];


% 例子
% qwe = []
% qwe(end+1).name = 'qwe';
% qwe(end).aug = augdict;


%% 加载病人信息
workspaces = load(strcat(block_mat_path,filesep,'subject',filesep,'subject.mat'));
subject = workspaces.subject;%记录所有统计信息的mat文件，为结构体数组。


sub_num_A = 0;%非PD数量
sub_num_B = 0;%PD数量
%给病人加上标签=======================================================================
for i = 1:length(subject)
    %根据样本命名规则，赋予样本对应的label
    %比如前20号是正样本，其余是负样本，实现如下
    if ismember(subject(i).id, augdict.class_a_id)
        label_1 = 0;label_2 = 0;label_3 = 1;%手动调整
        sub_num_A = sub_num_A+1;
    elseif ismember(subject(i).id, augdict.class_b_id)
        label_1 = 0;label_2 = 1;label_3 = 0;%手动调整
        sub_num_B = sub_num_B+1;
    end
    
    % 构建3分类独热编码
    subject(i).label = [label_1;label_2;label_3];
    
    % 构建2分类独热编码
    subject(i).label1 = [label_1;label_2];
    subject(i).label2 = [label_1;label_3];
    subject(i).label3 = [label_2;label_3];
    
    % 构建2分类普通label
    subject(i).label_1 = label_1;
    subject(i).label_2 = label_2;
    subject(i).label_3 = label_3;
end

%计算每一类样本扩增的总数量
aug_num_A = augdict.aug_num*(augdict.a_b_ratio(1)/sum(augdict.a_b_ratio));%A类样本扩增的总数量
aug_num_B = augdict.aug_num*(augdict.a_b_ratio(2)/sum(augdict.a_b_ratio));%B类样本扩增的总数量


%计算各类中每个样本扩增的总数量
per_sub_augnum_A = (aug_num_A/sub_num_A);%A类中每个样本扩增的大概数量
per_sub_augnum_B = (aug_num_B/sub_num_B);%B类中每个样本扩增的大概数量


%% 接下来以病人为单位进行操作
for i = 1:length(subject)
    subject_ins = subject(i);
    blocks = subject_ins.blocks_num_all;
    % 判断类别并计算该病人的每个block扩增的数量
    if ismember(subject(i).id, augdict.class_a_id)
        per_sub_augnum = per_sub_augnum_A;
    elseif  ismember(subject(i).id, augdict.class_b_id)
        per_sub_augnum = per_sub_augnum_B;
    end
    per_block_augnum = ceil(per_sub_augnum/blocks);
    
    %将信息储存到病人结构体数组中
    subject(i).per_block_aug_num = per_block_augnum;
    subject(i).per_tumor_aug_num = per_block_augnum*(subject(i).blocks_num_per_tumor);
    subject(i).all_aug_num = sum(subject(i).per_tumor_aug_num);
    
    
    %在病人为首要单位前提下,以每个block为次要单位进行扩增
    aug_count_num = 0;
    for ii = 1:subject(i).blocks_num_all
        data_path=strcat(block_mat_path,filesep,num2str(subject(i).id),'_',num2str(ii),'.mat');
        workspaces = load(data_path);
        data = workspaces.block;
        data = mat2gray(data);%一定要归一化，不然imadjust的时候会截断很大一部分值。！！！！！！！！
        
        if subject(i).othermode
            data_othermode = workspaces.block_othermode;
            data_othermode = mat2gray(data_othermode);%一定要归一化，不然imadjust的时候会截断很大一部分值。！！！！！！！！
        end
        
        
        
        
%         [aug_block,aug_block_othermode,augdict] = aug43D(block, aug_dict,othermode_flag,block_othermode)
        
        
        for iii = 1:per_block_augnum
            % 扩增
            if subject(i).othermode
                [aug_data,aug_data_othermode,aug_detail] = aug43D(data,  augdict,  subject(i).othermode,  data_othermode);
            else
                [aug_data,~                 ,aug_detail] = aug43D(data,  augdict,  subject(i).othermode,  []);
            end

            aug_count_num = aug_count_num + 1;
            
            % 保存 到 detail里面           
            tmp_name = strcat(num2str(subject(i).id),'_',num2str(ii),'_',num2str(aug_count_num));
            block_detail(end+1).name = tmp_name;
            block_detail(end).aug_detail = aug_detail;
            
            
            write_name = strcat(tmp_name,'.h5');
            fprintf('aug_num:%d    aug_file_name: %s\n',aug_count_num,write_name);
            finalpath = strcat(augdict.mat_savepath,filesep,write_name);%最终储存路径
            disp(finalpath);
            % 写入H5
            h5create(finalpath, '/data', size(aug_data),'Datatype','single');
            h5write(finalpath, '/data', aug_data);
                        
            if subject(i).othermode
                h5create(finalpath, '/data_othermode', size(aug_data_othermode),'Datatype','single');
                h5write(finalpath, '/data_othermode', aug_data_othermode);
            end
            
            
            h5create(finalpath, '/label_1', size(subject(i).label_1),'Datatype','single');
            h5write(finalpath, '/label_1', subject(i).label_1);
            h5create(finalpath, '/label_2', size(subject(i).label_2),'Datatype','single');
            h5write(finalpath, '/label_2', subject(i).label_2);
            h5create(finalpath, '/label_3', size(subject(i).label_3),'Datatype','single');
            h5write(finalpath, '/label_3', subject(i).label_3);
            h5create(finalpath, '/label', size(subject(i).label),'Datatype','single');
            h5write(finalpath, '/label', subject(i).label);
            h5create(finalpath, '/label1', size(subject(i).label1),'Datatype','single');
            h5write(finalpath, '/label1', subject(i).label1);
            h5create(finalpath, '/label2', size(subject(i).label2),'Datatype','single');
            h5write(finalpath, '/label2', subject(i).label2);
            h5create(finalpath, '/label3', size(subject(i).label3),'Datatype','single');
            h5write(finalpath, '/label3', subject(i).label3);
            
            
        end
    end
end


mkdir(strcat(augdict.mat_savepath,filesep,'subject'));
save(strcat(augdict.mat_savepath,filesep,'subject',filesep,'subject.mat'),'subject','augdict','block_detail');





































