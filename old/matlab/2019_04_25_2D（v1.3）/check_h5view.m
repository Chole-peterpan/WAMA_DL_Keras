%check
clear;
clc;




name= '27_1';
for i = 1:99
    aug_path = ['G:\@data_NENs_response\data_2D\3aug_20000\',name,'_',num2str(i),'.h5'];
    or_path = ['G:\@data_NENs_response\data_2D\4test\',name,'.h5'];
    
    original_img = h5read(or_path,'/data');
    olabel1=h5read(or_path,'/label_1');
    olabel2=h5read(or_path,'/label_2');
    olabel3=h5read(or_path,'/label_3');
    olabel=h5read(or_path,'/label');
    
    
    aug_img = h5read(aug_path,'/data');
    auglabel1=h5read(aug_path,'/label_1');
    auglabel2=h5read(aug_path,'/label_2');
    auglabel3=h5read(aug_path,'/label_3');
    auglabel=h5read(aug_path,'/label');
    
    % original_img = mat2gray(original_img);
    % aug_img = mat2gray(aug_img);
    
    
    
    www=figure;
    for iii=1:1
        imshowpair(original_img(:,:,iii), aug_img(:,:,iii), 'mon');
        title(num2str(iii));
        pause(0.8);
    end
    close(www);
    disp(['orlabel  :',num2str(olabel)']);
    disp(['auglabel  :',num2str(auglabel)']);
    
    
end







