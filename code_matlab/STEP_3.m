%% 生成未扩增的数据
clc;
clear;

%%
block_mat_path =  'H:\@data_NENs_recurrence\PNENs\data_outside\flow2\2block';
mat_savepath =    'H:\@data_NENs_recurrence\PNENs\data_outside\flow2\3or_out';
filename_list = dir(strcat(block_mat_path,filesep,'*.mat'));
augdict.savefomat.mode = 1;
augdict.savefomat.param = [150,150,80];


workspaces = load(strcat(block_mat_path,filesep,'subject',filesep,'subject.mat'));
subject = workspaces.subject;%记录所有统计信息的mat文件，为结构体数组。

%% 数据标签
% class_a_id = [[1:43],[46,47,49],[58,59]];% 手动传入a类病人的id
% class_b_id = 50:57;% 手动传入b类病人的id

class_a_id = 1:9;% 手动传入a类病人的id
class_b_id = 10:18;% 手动传入b类病人的id

%% 

for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    tmpcell = strsplit(filename,'_');
    id = str2double(tmpcell{1}(2:end));
    %比如前20号是正样本，其余是负样本，实现如下
    if ismember(id, class_a_id)
        label_1 = 0;label_2 = 0;label_3 = 1;%手动调整
    elseif ismember(id, class_b_id)
        label_1 = 0;label_2 = 1;label_3 = 0;%手动调整
    end
    
    % 构建3分类独热编码
    label = [label_1;label_2;label_3];
    
    % 构建2分类独热编码
    label1 = [label_1;label_2];
    label2 = [label_1;label_3];
    label3 = [label_2;label_3];
    
    
    % 读取block
    data_path=strcat(block_mat_path,filesep,filename);
    workspaces = load(data_path);
    data = workspaces.block;
    data = mat2gray(data);%一定要归一化，不然imadjust的时候会截断很大一部分值。！！！！！！！！
    
    if subject(1).othermode
        data_othermode = workspaces.block_othermode;
        data_othermode = mat2gray(data_othermode);%一定要归一化，不然imadjust的时候会截断很大一部分值。！！！！！！！！
    end

    % 处理block，使之shape相同
    if subject(1).othermode
        [aug_data,aug_data_othermode,~] = aug43D(data,  augdict,  subject(1).othermode,  data_othermode);
    else
        [aug_data,~                 ,~] = aug43D(data,  augdict,  subject(1).othermode,  []);
    end
    
    tmp_name = filename_list(ii,1).name;
    write_name = (tmp_name(1:end-4));


    finalpath = strcat(mat_savepath,filesep,write_name,'.h5');
    disp(finalpath);
    %----------------------------------------
    h5create(finalpath, '/data', size(aug_data),'Datatype','single');
    h5write(finalpath, '/data', aug_data);
    if subject(1).othermode
        h5create(finalpath, '/data_othermode', size(aug_data_othermode),'Datatype','single');
        h5write(finalpath, '/data_othermode', aug_data_othermode);
    end
    %------------------------------------------------
    h5create(finalpath, '/label_1', size(label_1),'Datatype','single');
    h5write(finalpath, '/label_1', label_1);
    
    h5create(finalpath, '/label_2', size(label_2),'Datatype','single');
    h5write(finalpath, '/label_2', label_2);
    
    h5create(finalpath, '/label_3', size(label_3),'Datatype','single');
    h5write(finalpath, '/label_3', label_3);
    
    h5create(finalpath, '/label1', size(label1),'Datatype','single');
    h5write(finalpath, '/label1', label1);
    
    h5create(finalpath, '/label2', size(label2),'Datatype','single');
    h5write(finalpath, '/label2', label2);
    
    h5create(finalpath, '/label3', size(label3),'Datatype','single');
    h5write(finalpath, '/label3', label3);
    
    h5create(finalpath, '/label', size(label),'Datatype','single');
    h5write(finalpath, '/label', label);

end











