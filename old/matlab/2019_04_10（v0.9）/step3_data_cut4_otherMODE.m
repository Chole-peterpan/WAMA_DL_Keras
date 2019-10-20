%���Ѿ����˶�����cut��ִ���������������cut��ͬʱ�������������ݲ���Ҳͳһ��
%ע���������򣬶����ڱ��1��100�������ڱ����Ϊ��Ӧ���+100*n��n����ʵ�����󶨡�

clc;
clear;
mat_path = 'G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@data\v\2pre';
mat_savepath = 'G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@data\v\3cut';

%�ο���������·��������׼��ͳһ������resize��refmode·�������ݶ�Ӧ�Ĳ���
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
    
    %��ȡ�ο�cut����ĺ��
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











