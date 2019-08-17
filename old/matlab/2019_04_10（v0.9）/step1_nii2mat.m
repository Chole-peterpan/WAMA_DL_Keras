%完成数据的格式转换，并将CT和掩膜储存到一个mat中

clc;
clear;
nii_path = 'G:\@data_zhuanyi_zhongshan\PNENs\CT\v';
gt_path = 'G:\@data_zhuanyi_zhongshan\PNENs\mask\v';
mat_savepath = 'G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@data\v\1mat';

nii_filename_list = dir(strcat(nii_path,filesep,'*.nii'));
for ii = 1:length(nii_filename_list)
    filename = nii_filename_list(ii,1).name;
    Vref = spm_vol(strcat(nii_path,filesep,filename));
    Data = spm_read_vols(Vref);

    numm=str2double(filename(1:end-4));
    Vref_g = spm_vol(strcat(gt_path,filesep,num2str(numm-100),'.nii'));
    GT_mask = spm_read_vols(Vref_g);
    
    save(strcat(mat_savepath,filesep,filename(1:end-4)),'Data','GT_mask');   
    disp(filename);
end 











