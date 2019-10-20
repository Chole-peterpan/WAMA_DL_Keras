clc;
clear;
mat_path = 'G:\@data_zhuanyi_zhongshan\PNENs_wang\2pre';
mat_savepath = 'G:\@data_zhuanyi_zhongshan\PNENs_wang\3cut';
mat_savepath_keepsize = 'G:\@data_zhuanyi_zhongshan\PNENs_wang\3cut_keepsize';

filename_list = dir(strcat(mat_path,filesep,'*.mat'));

%% cut and resize
for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    data = workspaces.Data;
    mask = workspaces.mask;
    index = find(mask);

    % cut
    [I1,I2,I3] = ind2sub(size(mask),index);
    data_cut = data(min(I1):max(I1),min(I2):max(I2),min(I3):max(I3));
    mask_cut = mask(min(I1):max(I1),min(I2):max(I2),min(I3):max(I3));
    or_size = size(data_cut);
    
    
    %resize
    Data_cu = imresize3(data_cut,[280,280,size(data_cut,3)],'cubic');
    mask_cu = imresize3(mask_cut,[280,280,size(data_cut,3)],'cubic');
    %  mask is wrong
    
    save(strcat(mat_savepath,filesep,filename(1:end-4)),'Data_cu','mask_cu','or_size');  
end


%% cut and keep size
for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    data = workspaces.Data;
    mask = workspaces.mask;
    index = find(mask);

    % cut
    [I1,I2,I3] = ind2sub(size(mask),index);
    data_cut = data(min(I1):max(I1),min(I2):max(I2),min(I3):max(I3));
    mask_cut = mask(min(I1):max(I1),min(I2):max(I2),min(I3):max(I3));
    or_size = size(data_cut);
    
    if or_size(1)>280 || or_size(2)>280
        %resize
        data_cut = imresize3(data_cut,[280,280,size(data_cut,3)],'cubic');
        mask_cut = imresize3(mask_cut,[280,280,size(data_cut,3)],'cubic');
        or_size = size(data_cut);
    end
    
    %putin
    Data_cu = zeros(280,280,or_size(3));
    mask_cu = zeros(280,280,or_size(3));
    
    indexx1 = 140-floor(or_size(1)/2);
    indexx2 = 140-floor(or_size(2)/2);
    Data_cu(indexx1+1:indexx1+or_size(1),indexx2+1:indexx2+or_size(2),:) = data_cut;
    mask_cu(indexx1+1:indexx1+or_size(1),indexx2+1:indexx2+or_size(2),:) = mask_cut;

    save(strcat(mat_savepath_keepsize,filesep,filename(1:end-4)),'Data_cu','mask_cu','or_size');  
end








