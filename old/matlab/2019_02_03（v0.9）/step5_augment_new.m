clc;
clear;
mat_path = 'G:\$code_dasheng\final_data\@new_data\4block\fufa';%block
mat_savepath = 'H:\@data_dasheng_fufa_dl\@aug_data';
filename_list = dir(strcat(mat_path,filesep,'*.mat'));

subject_path = 'G:\$code_dasheng\final_data\@new_data\1mat\fufa';%仅仅需要对应类别样本数量，1mat即可，不要block文件夹
sub_list = dir(strcat(subject_path,filesep,'*.mat'));
%% 焕辉数据nens标签
% % G1
% label_1 = 1;label_2 = 0;label_3 = 0;
% % G2
% label_1 = 0;label_2 = 1;label_3 = 0;
% % G3
% label_1 = 0;label_2 = 0;label_3 = 1;
%% 大盛数据肝癌标签
%liver fufa
label_1 = 0;label_2 = 0;label_3 = 1;
% %liver bufufa
% label_1 = 0;label_2 = 1;label_3 = 0;
%% pnens肝转移数据标签
% % zhuanyi
% label_1 = 0;label_2 = 0;label_3 = 1;
% % % buzhuanyi
% % label_1 = 0;label_2 = 1;label_3 = 0;
%% 
label = [label_1;label_2;label_3];


%% aug
aug_num = 25000;%自己设定每类样本扩增的总数量，最终扩增数量会接近此数量
sub_num = length(sub_list);
per_sub_augnum = floor(aug_num/sub_num);

for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    data = workspaces.Data;
    blocks = workspaces.blocks;
    
    %% 根据块数量确定每个块需要扩增多少
    per_block_augnum = (per_sub_augnum/blocks)/2;%每个块需要扩增的数量,这里需要注意，因为有翻转，所以这个需要再除以2    
    %再根据上面的块扩增数量，确定扩增的参数，旋转角度范围及步长，
    %三个扩增范围，也就是三个数量相乘，设定每种扩增方法扩增数量相同，则需要开三次方
    per_method_num = floor(nthroot(per_block_augnum,3)); 
    %ad那两个参数是对比度调节的参数（lowin和highin，因为图像已经归一化了，所以lowin在0到0.1中间取，highin在0.8到1之间取）
    ad2_1 = 0.800; ad2_2 = 1.00;  ad2_step = (ad2_2-ad2_1)/(per_method_num-1);%需要-1，不然会使扩增数量增加               
    ad1_1 = 0;     ad1_2 = 0.1;   ad1_step = (ad1_2-ad1_1)/(per_method_num-1);         
    r_1   = 0;     r_2   = 40;    r_step   = (r_2-r_1)/(per_method_num-1);             
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
%     disp(['final aug num is:',num2str(final_num*2)]);
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
    disp(['final aug num is:',num2str(final_num*2)]);
    
    
    
    %%  aug
    aug_count_num = 0;
    image_roi_output = zeros(size(data));
    %原方向
    for angel = r_1:r_step:r_2
        for m = ad1_1:ad1_step:ad1_2
            for n = ad2_1:ad2_step:ad2_2
                fprintf('%f %f %f\n',angel,m,n);
                for i = 1:16
                    image_roi_input= data(:,:,i);
                    image_roi_rotate_o = imrotate(image_roi_input,angel,'bilinear','crop');
                    image_roi_adjust = imadjust(image_roi_rotate_o,[m n],[]);
                    image_roi_output(:,:,i) =  image_roi_adjust;
                end
                %旋转之后因为插值会产生大于1的数字，所以最后需要归一化
                image_roi_output = mat2gray(image_roi_output);
                %save
                aug_count_num = aug_count_num + 1;
                tmp_name = filename_list(ii,1).name;
                write_name = strcat(tmp_name(1:end-4),'_',num2str(aug_count_num));
                fprintf('%d %s\n',aug_count_num,strcat(mat_path,filesep,filename));
                %                 image_roi_output = permute(image_roi_output,[3 2 1]);
                finalpath = strcat(mat_savepath,filesep,write_name,'.h5');
                disp(finalpath);
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
                %                 image_roi_output = permute(image_roi_output,[3 2 1]);
            end
        end
    end
    %翻转=====================================================================================
    for angel = r_1:r_step:r_2
        for m = ad1_1:ad1_step:ad1_2
            for n = ad2_1:ad2_step:ad2_2
                fprintf('%f %f %f\n',angel,m,n);
                for i = 1:16
                    image_roi_input= fliplr(data(:,:,i));
                    image_roi_rotate_o = imrotate(image_roi_input,angel,'bilinear','crop');
                    image_roi_adjust = imadjust(image_roi_rotate_o,[m n],[]);
                    image_roi_output(:,:,i) =  image_roi_adjust;
                end
                %旋转之后因为插值会产生大于1的数字，所以最后需要归一化
                %不过归一化之后是否会影响之前标准化的效果呢，暂时不知道
                image_roi_output = mat2gray(image_roi_output);
                %save
                aug_count_num = aug_count_num + 1;
                tmp_name = filename_list(ii,1).name;
                write_name = strcat(tmp_name(1:end-4),'_',num2str(aug_count_num));
                fprintf('%d %s\n',aug_count_num,strcat(mat_path,filesep,filename));
                %                 image_roi_output = permute(image_roi_output,[3 2 1]);
                finalpath = strcat(mat_savepath,filesep,write_name,'.h5');
                disp(finalpath);
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
                %                 image_roi_output = permute(image_roi_output,[3 2 1]);
            end
        end
    end
    
    
    
end


% % view
% qwe=data(:,:,1);
% qweqwe=imadjust(qwe,[0.1 0.8],[]);
% qweqweqwe=adapthisteq(qwe);%defaut param
% qweqweqweqwe=adapthisteq(qwe, 'NumTiles', [25 25]);%defaut param
% qweqweqweqweqwe = fliplr(qwe);
% figure;imshowpair(qwe,qweqwe, 'mon');title('1');
% figure;imshowpair(qwe,qweqweqwe, 'mon');title('2');
% figure;imshowpair(qwe,qweqweqweqwe, 'mon');title('3');
% figure;imshowpair(qwe,qweqweqweqweqwe, 'mon');title('4');
% 
% 
% image_rotate_o = imrotate(qwe,15,'bilinear','crop');
% image_rotate_o = mat2gray(image_rotate_o);
% %旋转之后因为插值会产生大于1的数字，所以最后需要归一化
% figure;imshowpair(qwe,image_rotate_o, 'mon');title('4');
% max(max(image_rotate_o))









