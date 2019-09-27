%% ����δ����������
clc;
clear;

%%
block_mat_path =  'H:\@data_NENs_recurrence\PNENs\data_outside\flow2\2block';
mat_savepath =    'H:\@data_NENs_recurrence\PNENs\data_outside\flow2\3or_out';
filename_list = dir(strcat(block_mat_path,filesep,'*.mat'));
augdict.savefomat.mode = 1;
augdict.savefomat.param = [150,150,80];


workspaces = load(strcat(block_mat_path,filesep,'subject',filesep,'subject.mat'));
subject = workspaces.subject;%��¼����ͳ����Ϣ��mat�ļ���Ϊ�ṹ�����顣

%% ���ݱ�ǩ
% class_a_id = [[1:43],[46,47,49],[58,59]];% �ֶ�����a�ಡ�˵�id
% class_b_id = 50:57;% �ֶ�����b�ಡ�˵�id

class_a_id = 1:9;% �ֶ�����a�ಡ�˵�id
class_b_id = 10:18;% �ֶ�����b�ಡ�˵�id

%% 

for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    tmpcell = strsplit(filename,'_');
    id = str2double(tmpcell{1}(2:end));
    %����ǰ20�����������������Ǹ�������ʵ������
    if ismember(id, class_a_id)
        label_1 = 0;label_2 = 0;label_3 = 1;%�ֶ�����
    elseif ismember(id, class_b_id)
        label_1 = 0;label_2 = 1;label_3 = 0;%�ֶ�����
    end
    
    % ����3������ȱ���
    label = [label_1;label_2;label_3];
    
    % ����2������ȱ���
    label1 = [label_1;label_2];
    label2 = [label_1;label_3];
    label3 = [label_2;label_3];
    
    
    % ��ȡblock
    data_path=strcat(block_mat_path,filesep,filename);
    workspaces = load(data_path);
    data = workspaces.block;
    data = mat2gray(data);%һ��Ҫ��һ������Ȼimadjust��ʱ���ضϺܴ�һ����ֵ������������������
    
    if subject(1).othermode
        data_othermode = workspaces.block_othermode;
        data_othermode = mat2gray(data_othermode);%һ��Ҫ��һ������Ȼimadjust��ʱ���ضϺܴ�һ����ֵ������������������
    end

    % ����block��ʹ֮shape��ͬ
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











