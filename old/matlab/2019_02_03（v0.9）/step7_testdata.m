clc;
clear;
mat_path = 'G:\$code_dasheng\final_data\@new_data\4block\bufufa';
mat_savepath = 'H:\@data_dasheng_fufa_dl\@test_data';
filename_list = dir(strcat(mat_path,filesep,'*.mat'));
%% 
% % G1
% label_1 = 1;label_2 = 0;label_3 = 0;
% % G2
% label_1 = 0;label_2 = 1;label_3 = 0;
% % G3
% label_1 = 0;label_2 = 0;label_3 = 1;

%% 
% %liver fufa
% label_1 = 0;label_2 = 0;label_3 = 1;
%liver bufufa
label_1 = 0;label_2 = 1;label_3 = 0;


%% pnens肝转移数据标签
% % zhuanyi
% label_1 = 0;label_2 = 0;label_3 = 1;
% % buzhuanyi
% label_1 = 0;label_2 = 1;label_3 = 0;


label = [label_1;label_2;label_3];

for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    data = workspaces.Data;
    %mask = workspaces.Mask; %没有用到

    
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
    
    
    
end











