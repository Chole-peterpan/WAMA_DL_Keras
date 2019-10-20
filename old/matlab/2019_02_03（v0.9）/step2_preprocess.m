clc;
clear;
mat_path = 'G:\@data_zhuanyi_zhongshan\PNENs_wang\1mat';
mat_savepath = 'G:\@data_zhuanyi_zhongshan\PNENs_wang\2pre';

nii_filename_list = dir(strcat(mat_path,filesep,'*.mat'));


for ii = 1:length(nii_filename_list)
    filename = nii_filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    data = workspaces.Data;
    mask = workspaces.GT_mask;
    [xx,yy,zz]=size(data);
    
    % adjust CT window
    data(data<-30)=-30;
    data(data>280)=280;
    
    % standardization
    data_flatten=reshape(data,[1,xx*yy*zz]);
    meann = mean(data_flatten); % do not code like this :meann = mean(mean(mean(data)));
    stdd = std(data_flatten);  % do not code like this :stdd = std(std(std(data)));
    data_nr = (data-meann)/stdd; 
    
    % standardization(method 2)===================
%     data_nr1 = zscore(data_flatten);
%     data_nr1 = reshape(data_nr1,size(data));

    % normolization
    minn = min(min(min(data_nr)));
    maxx = max(max(max(data_nr)));
    data_no = (data_nr - minn)/(maxx - minn);
    
    % normolization (method 2)====================
%     data_no1 = mat2gray(data_nr);
    
    Data = data_no;
    save(strcat(mat_savepath,filesep,filename(1:end-4)),'Data','mask');   
end 



















