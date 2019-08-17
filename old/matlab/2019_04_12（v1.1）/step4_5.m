%扩增原图
%需要先将test集合跑出，之后重新命名罢了,故实际应该是step4_5

clc;
clear;
h5_path = 'H:\@data_liaoxiao\4test';
h5_savepath = 'H:\@data_liaoxiao\3aug_or';
filename_list = dir(strcat(h5_path,filesep,'*.h5'));






for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    full_path = strcat(h5_path,filesep,filename);
    
    image_roi_output = h5read(full_path,'/data');
    label_1=h5read(full_path,'/label_1');
    label_2=h5read(full_path,'/label_2');
    label_3=h5read(full_path,'/label_3');
    label=h5read(full_path,'/label');
    label1=h5read(full_path,'/label1');
    label2=h5read(full_path,'/label2');
    label3=h5read(full_path,'/label3');
    
    
    finalpath = strcat(h5_savepath,filesep,filename(1:end-3),'_1.h5');
    
    
        
    h5create(finalpath, '/data', size(image_roi_output),'Datatype','single');
    h5write(finalpath, '/data', image_roi_output);
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