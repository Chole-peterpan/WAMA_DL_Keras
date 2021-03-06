%初始化
clc;
clear;
%参数设置
block_mat_path = 'H:\@data_liaoxiao\2block';%block
mat_savepath = 'H:\@data_liaoxiao\testqweqwe';
aug_num = 6000;%总扩增数量
a_b_ratio = [1,2];%例如AB比例为1:2，则设置为[1,2],A是非PD，B是PD

%==========================================================================
aug_num_A = aug_num*(a_b_ratio(1)/sum(a_b_ratio));%A类样本扩增的总数量
aug_num_B = aug_num*(a_b_ratio(2)/sum(a_b_ratio));%B类样本扩增的总数量
%加载病人信息
workspaces = load(strcat(block_mat_path,filesep,'subject_all',filesep,'subject.mat'));
subject_all = workspaces.subject_all;%记录所有统计信息的mat文件，为结构体数组。
%给病人加上标签=======================================================================
for i = 1:length(subject_all)
    %根据样本命名规则，赋予样本对应的label
    %比如前20号是正样本，其余是负样本，实现如下
    if subject_all(i).id <= 20 %非PD
        label_1 = 0;label_2 = 0;label_3 = 1;%手动调整
    else %PD
        label_1 = 0;label_2 = 1;label_3 = 0;%手动调整
    end
    subject_all(i).label = [label_1;label_2;label_3];
    subject_all(i).label1 = [label_1;label_2];
    subject_all(i).label2 = [label_1;label_3];
    subject_all(i).label3 = [label_2;label_3];
    
    subject_all(i).label_1 = label_1;
    subject_all(i).label_2 = label_2;
    subject_all(i).label_3 = label_3;
end


%% 开始扩增wdnmd
sub_num_A = 0;%非PD数量
sub_num_B = 0;%PD数量
for i = 1:length(subject_all)
    if subject_all(i).label(3) == 1 %非PD
        sub_num_A = sub_num_A+1;
    elseif  subject_all(i).label(2) == 1 %PD
        sub_num_B = sub_num_B+1;
    end
end

per_sub_augnum_A = floor(aug_num_A/sub_num_A);%非PD 每个样本扩增的大概数量
per_sub_augnum_B = floor(aug_num_B/sub_num_B);%PD 每个样本扩增的大概数量



for i = 1:length(subject_all)
    subject = subject_all(i);
    blocks = subject.blocks_num_all;
    % 判断类别
    if subject.label(3) == 1 %非PD
        per_sub_augnum = per_sub_augnum_A;
    elseif  subject.label(2) == 1 %PD
        per_sub_augnum = per_sub_augnum_B;
    end
    
    per_block_augnum = (per_sub_augnum/blocks)/2;
    %per_block_augnum为每个块需要扩增的数量,这里需要注意，因为有翻转，所以这个需要再除以2
    %再根据上面的块扩增数量，确定扩增的参数，旋转角度范围及步长，
    %三个扩增范围，也就是三个数量相乘，设定每种扩增方法扩增数量相同，则需要开三次方 
    per_method_num = floor(nthroot(per_block_augnum,3)); %计算每种扩增方法扩增的次数（设定每种方法扩增的次数相同）
    %ad那两个参数是对比度调节的参数（lowin和highin，因为图像已经归一化了，所以lowin在0到0.1中间取，highin在0.8到1之间取）
    ad2_1 = 0.800; ad2_2 = 1.00;  ad2_step = abs((ad2_2-ad2_1)/(per_method_num-1));%需要-1，不然会使扩增数量增加               
    ad1_1 = 0;     ad1_2 = 0.1;   ad1_step = abs((ad1_2-ad1_1)/(per_method_num-1));         
    r_1   = 0;     r_2   = 40;    r_step   = abs((r_2-r_1)/(per_method_num-1));  
    %步长一定要加绝对值，不然当数量过小的时候，会出现负数步长的情况没这样会导致扩增数量为0
    %初步计算最后扩增数量
    flag1 = 0;
    for angel = r_1:r_step:r_2
        flag1 = flag1+1;
    end
    flag2 = 0;
    for highh = ad2_1:ad2_step:ad2_2
        flag2 = flag2+1;
    end
    flag3 = 0;
    for loww = ad1_1:ad1_step:ad1_2
        flag3 = flag3+1;
    end
    final_num = flag1*flag2*flag3;
    %=====================================================================================================
    % 矫正：如果扩增数量与设定数量差距太大，则迭代适当减小步长（以下代码针对于角度），直到差距不断减小到可接受范围
    % ps：以下只针对一个参数矫正，也可以针对多个，即逐级进行矫正，误差0.2级别的先利用一个参数矫正一次，之后误差减小到0.005级别则再用另外一个参数矫正
    r_num  =per_method_num-1;
    for iiqweqwe = 1:99
        %刚开始一定是小于设定数量的，所以如果差距比例大于0.05，则增大数量
        if (abs(per_block_augnum-final_num)/per_block_augnum)>0.05
            r_num=r_num+1;
        end
        %计算增大数量后的总扩增数量
        r_step   = (r_2-r_1)/(r_num);
        flag1 = 0;
        for angel = r_1:r_step:r_2
            flag1 = flag1+1;
        end
        final_num = flag1*flag2*flag3;
        %判断，如果小于0.05则break，如果大于设定数量也break
        if (abs(per_block_augnum-final_num)/per_block_augnum)<=0.05  || final_num>=per_block_augnum
            break;
        end 
    end
    disp(['final block aug num is:',num2str(final_num*2)]);
    
    
    %将block信息储存到病人结构体数组中
    subject_all(i).per_block_aug_num = final_num*2;
    subject_all(i).per_tumor_aug_num = final_num*2*(subject_all(i).blocks_num_per_tumor);
    subject_all(i).all_aug_num = sum(subject_all(i).per_tumor_aug_num);
    
    
    %%  aug 在病人单位下,以每个block为单位进行扩增
    aug_count_num = 0;
    data_path=strcat(block_mat_path,filesep,num2str(subject_all(1).id),'_',num2str(1),'.mat');
    workspaces = load(data_path);
    data = workspaces.block;
    image_roi_output = zeros(size(data));
    
    for ii = 1:subject_all(i).blocks_num_all
        data_path=strcat(block_mat_path,filesep,num2str(subject_all(i).id),'_',num2str(ii),'.mat');
        workspaces = load(data_path);
        data = workspaces.block;
        data = mat2gray(data);%一定要归一化，不然imadjust的时候会截断很大一部分值。！！！！！！！！
        %原方向
        for angel = r_1:r_step:r_2
            for m = ad1_1:ad1_step:ad1_2
                for n = ad2_1:ad2_step:ad2_2
                    fprintf('%f %f %f\n',angel,m,n);
                    for iiii = 1:16
                        image_roi_input= data(:,:,iiii);
                        image_roi_rotate_o = imrotate(image_roi_input,angel,'bilinear','crop');
                        image_roi_adjust = imadjust(image_roi_rotate_o,[m n],[]);
                        image_roi_output(:,:,iiii) =  image_roi_adjust;
                    end
                    %旋转之后因为插值会产生大于1的数字，所以最后需要归一化,不过之前已经标准化了，所以也可不用归一化。
                    image_roi_output = mat2gray(image_roi_output);%一定要归一化,test集合也要归一化
                    %save
                    aug_count_num = aug_count_num + 1;
                    tmp_name = strcat(num2str(subject_all(i).id),'_',num2str(ii),'_',num2str(aug_count_num));
                    write_name = strcat(tmp_name,'.h5');
                    fprintf('aug_num:%d    aug_file_name: %s\n',aug_count_num,write_name);
                    %image_roi_output = permute(image_roi_output,[3 2 1]);
                    finalpath = strcat(mat_savepath,filesep,write_name);
                    disp(finalpath);
                    h5create(finalpath, '/data', size(image_roi_output),'Datatype','single');
                    h5write(finalpath, '/data', image_roi_output);
                    h5create(finalpath, '/label_1', size(subject_all(i).label_1),'Datatype','single');
                    h5write(finalpath, '/label_1', subject_all(i).label_1);
                    h5create(finalpath, '/label_2', size(subject_all(i).label_2),'Datatype','single');
                    h5write(finalpath, '/label_2', subject_all(i).label_2);
                    h5create(finalpath, '/label_3', size(subject_all(i).label_3),'Datatype','single');
                    h5write(finalpath, '/label_3', subject_all(i).label_3);
                    h5create(finalpath, '/label', size(subject_all(i).label),'Datatype','single');
                    h5write(finalpath, '/label', subject_all(i).label);
                    h5create(finalpath, '/label1', size(subject_all(i).label1),'Datatype','single');
                    h5write(finalpath, '/label1', subject_all(i).label1);
                    h5create(finalpath, '/label2', size(subject_all(i).label2),'Datatype','single');
                    h5write(finalpath, '/label2', subject_all(i).label2);
                    h5create(finalpath, '/label3', size(subject_all(i).label3),'Datatype','single');
                    h5write(finalpath, '/label3', subject_all(i).label3);
                    %                 image_roi_output = permute(image_roi_output,[3 2 1]);
                end
            end
        end
        %翻转=====================================================================================
        for angel = r_1:r_step:r_2
            for m = ad1_1:ad1_step:ad1_2
                for n = ad2_1:ad2_step:ad2_2
                    fprintf('%f %f %f\n',angel,m,n);
                    for iiii = 1:16
                        image_roi_input= fliplr(data(:,:,iiii));
                        image_roi_rotate_o = imrotate(image_roi_input,angel,'bilinear','crop');
                        image_roi_adjust = imadjust(image_roi_rotate_o,[m n],[]);
                        image_roi_output(:,:,iiii) =  image_roi_adjust;
                    end
                    %旋转之后因为插值会产生大于1的数字，所以最后需要归一化
                    %不过归一化之后是否会影响之前标准化的效果呢，暂时不知道
                    image_roi_output = mat2gray(image_roi_output);%一定要归一化,test集合也要归一化
                    %save
                    aug_count_num = aug_count_num + 1;
                    tmp_name = strcat(num2str(subject_all(i).id),'_',num2str(ii),'_',num2str(aug_count_num));
                    write_name = strcat(tmp_name,'.h5');
                    fprintf('aug_num:%d    aug_file_name: %s\n',aug_count_num,write_name);
                    %image_roi_output = permute(image_roi_output,[3 2 1]);
                    finalpath = strcat(mat_savepath,filesep,write_name);
                    disp(finalpath);
                    h5create(finalpath, '/data', size(image_roi_output),'Datatype','single');
                    h5write(finalpath, '/data', image_roi_output);
                    h5create(finalpath, '/label_1', size(subject_all(i).label_1),'Datatype','single');
                    h5write(finalpath, '/label_1', subject_all(i).label_1);
                    h5create(finalpath, '/label_2', size(subject_all(i).label_2),'Datatype','single');
                    h5write(finalpath, '/label_2', subject_all(i).label_2);
                    h5create(finalpath, '/label_3', size(subject_all(i).label_3),'Datatype','single');
                    h5write(finalpath, '/label_3', subject_all(i).label_3);
                    h5create(finalpath, '/label', size(subject_all(i).label),'Datatype','single');
                    h5write(finalpath, '/label', subject_all(i).label);
                    h5create(finalpath, '/label1', size(subject_all(i).label1),'Datatype','single');
                    h5write(finalpath, '/label1', subject_all(i).label1);
                    h5create(finalpath, '/label2', size(subject_all(i).label2),'Datatype','single');
                    h5write(finalpath, '/label2', subject_all(i).label2);
                    h5create(finalpath, '/label3', size(subject_all(i).label3),'Datatype','single');
                    h5write(finalpath, '/label3', subject_all(i).label3);
                    %                 image_roi_output = permute(image_roi_output,[3 2 1]);
                end
            end
        end
  
    end

    
end


mkdir(strcat(mat_savepath,filesep,'subject_all'));
save(strcat(mat_savepath,filesep,'subject_all',filesep,'subject.mat'),'subject_all'); 





