%��Ҫ�ֶ��ֿ��������ݼ�
clc;
clear;
mat_path = 'G:\@data_NENs_response\data_2D\2block\b';
mat_savepath = 'G:\@data_NENs_response\data_2D\4test';
filename_list = dir(strcat(mat_path,filesep,'*.mat'));


%% ���ݱ�ǩ
% % PD
label_1 = 0;label_2 = 1;label_3 = 0;
% % % ��PD
% label_1 = 0;label_2 = 0;label_3 = 1;


%��ʢ��hcc
% % a��1~21
% label_1 = 0;label_2 = 0;label_3 = 1;%�ֶ�����
% % b��22~41
% label_1 = 0;label_2 = 1;label_3 = 0;%�ֶ�����
%% 
label1 = [label_1;label_2];
label2 = [label_1;label_3];
label3 = [label_2;label_3];
label = [label_1;label_2;label_3];



for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    data = workspaces.block;
    %mask = workspaces.Mask; %û���õ�
    
    %һ��Ҫ��һ��,test����ҲҪ��һ��
    data = mat2gray(data);%һ��Ҫ��һ��,test����ҲҪ��һ��
    
    tmp_name = filename_list(ii,1).name;
    write_name = (tmp_name(1:end-4));

    %                 image_roi_output = permute(image_roi_output,[3 2 1]);
    finalpath = strcat(mat_savepath,filesep,write_name,'.h5');
    disp(finalpath);
    h5create(finalpath, '/data', size(data),'Datatype','single');
    h5write(finalpath, '/data', data);
    h5create(finalpath, '/label_1', size(label_1),'Datatype','single');
    h5write(finalpath, '/label_1', label_1);
    h5create(finalpath, '/label_2', size(label_2),'Datatype','single');
    h5write(finalpath, '/label_2', label_2);
    h5create(finalpath, '/label_3', size(label_3),'Datatype','single');
    h5write(finalpath, '/label_3', label_3);
    h5create(finalpath, '/label', size(label),'Datatype','single');
    h5write(finalpath, '/label', label);
    h5create(finalpath, '/label1', size(label1),'Datatype','single');
    h5write(finalpath, '/label1', label1);
    h5create(finalpath, '/label2', size(label2),'Datatype','single');
    h5write(finalpath, '/label2', label2);
    h5create(finalpath, '/label3', size(label3),'Datatype','single');
    h5write(finalpath, '/label3', label3);
    
    
end











