%当已经有了动脉期cut后，执行这个，将静脉期cut的同时，将两个期数据层数也统一。
%注意命名规则，动脉期编号1到100，静脉期编号则为对应编号+100*n，n根据实际需求定。

clc;
clear;
mat_path = 'G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@data\v\2pre';
mat_savepath = 'G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@data\v\3cut';

%参考数据所在路径（粗配准，统一将层数resize到refmode路径下数据对应的层数
mat_refmode = 'G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@data\a\3cut';
% mat_savepath_keepsize = 'G:\@data_zhuanyi_zhongshan\PNENs_wang\3cut_keepsize';

filename_list = dir(strcat(mat_path,filesep,'*.mat'));
name_res = 300;
%% cut and resize
for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    data = workspaces.Data;
    mask = workspaces.mask;
    index = find(mask);
    
    %读取参考cut矩阵的厚度
    numm=str2double(filename(1:end-4));
    workspaces_re = load(strcat(mat_refmode,filesep,num2str(numm-name_res),'.mat'));
    re_thickness = workspaces_re.or_size(3);
    
    % cut
    [I1,I2,I3] = ind2sub(size(mask),index);
    data_cut = data(min(I1):max(I1),min(I2):max(I2),min(I3):max(I3));
    mask_cut = mask(min(I1):max(I1),min(I2):max(I2),min(I3):max(I3));
    or_size = size(data_cut);
      
    %resize
    Data_cu = imresize3(data_cut,[280,280,re_thickness],'cubic');
    mask_cu = imresize3(mask_cut,[280,280,re_thickness],'cubic');
    %  mask is wrong
    
    save(strcat(mat_savepath,filesep,filename(1:end-4)),'Data_cu','mask_cu','or_size');  
end











