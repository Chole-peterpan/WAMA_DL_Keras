%check
clear;
clc;
data_path = 'F:\@data_pnens_zhuanyi_dl\aug';
oppath = 'G:\$code_huanhui\@data_huanhui\@mat_data\G1\2.mat';
filename_list = dir(strcat(data_path,filesep,'*.h5'));
workspaces = load(oppath);
data = workspaces.Data;



for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    tmppath = strcat(data_path,filesep,filename);
    original_img = h5read(tmppath,'/data');
    label1=h5read(tmppath,'/label_1');
    label2=h5read(tmppath,'/label_2');
    label3=h5read(tmppath,'/label_3');
    label=h5read(tmppath,'/label');
    
    www=figure;
    for iii=1:16
        imshowpair(data(:,:,iii), original_img(:,:,iii), 'mon');
        title(strcat(filename,':   :',num2str(iii)));
        pause(0.05);
    end
    close(www);
    
    
end


  
%% 
clear;
clc;
tmppath = 'H:\@data_pnens_zhuanyi_dl\aug\a\1_1_1.h5';
label_1=h5read(tmppath,'/label_1');
label_2=h5read(tmppath,'/label_2');
label_3=h5read(tmppath,'/label_3');
label=h5read(tmppath,'/label');
label1=h5read(tmppath,'/label1');
label2=h5read(tmppath,'/label2');
label3=h5read(tmppath,'/label3');


