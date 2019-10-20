clc;
clear;
mat_path = 'G:\$code_dasheng\final_data\@new_data\4block\fufa';%block
mat_savepath = 'H:\@data_dasheng_fufa_dl\@aug_data';
filename_list = dir(strcat(mat_path,filesep,'*.mat'));

subject_path = 'G:\$code_dasheng\final_data\@new_data\1mat\fufa';%������Ҫ��Ӧ�������������1mat���ɣ���Ҫblock�ļ���
sub_list = dir(strcat(subject_path,filesep,'*.mat'));
%% ��������nens��ǩ
% % G1
% label_1 = 1;label_2 = 0;label_3 = 0;
% % G2
% label_1 = 0;label_2 = 1;label_3 = 0;
% % G3
% label_1 = 0;label_2 = 0;label_3 = 1;
%% ��ʢ���ݸΰ���ǩ
%liver fufa
label_1 = 0;label_2 = 0;label_3 = 1;
% %liver bufufa
% label_1 = 0;label_2 = 1;label_3 = 0;
%% pnens��ת�����ݱ�ǩ
% % zhuanyi
% label_1 = 0;label_2 = 0;label_3 = 1;
% % % buzhuanyi
% % label_1 = 0;label_2 = 1;label_3 = 0;
%% 
label = [label_1;label_2;label_3];


%% aug
aug_num = 25000;%�Լ��趨ÿ����������������������������������ӽ�������
sub_num = length(sub_list);
per_sub_augnum = floor(aug_num/sub_num);

for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    data = workspaces.Data;
    blocks = workspaces.blocks;
    
    %% ���ݿ�����ȷ��ÿ������Ҫ��������
    per_block_augnum = (per_sub_augnum/blocks)/2;%ÿ������Ҫ����������,������Ҫע�⣬��Ϊ�з�ת�����������Ҫ�ٳ���2    
    %�ٸ�������Ŀ�����������ȷ�������Ĳ�������ת�Ƕȷ�Χ��������
    %����������Χ��Ҳ��������������ˣ��趨ÿ��������������������ͬ������Ҫ�����η�
    per_method_num = floor(nthroot(per_block_augnum,3)); 
    %ad�����������ǶԱȶȵ��ڵĲ�����lowin��highin����Ϊͼ���Ѿ���һ���ˣ�����lowin��0��0.1�м�ȡ��highin��0.8��1֮��ȡ��
    ad2_1 = 0.800; ad2_2 = 1.00;  ad2_step = (ad2_2-ad2_1)/(per_method_num-1);%��Ҫ-1����Ȼ��ʹ������������               
    ad1_1 = 0;     ad1_2 = 0.1;   ad1_step = (ad1_2-ad1_1)/(per_method_num-1);         
    r_1   = 0;     r_2   = 40;    r_step   = (r_2-r_1)/(per_method_num-1);             
    %�������������������
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
    % ��������������������趨�������̫��������ʵ���С���������´�������ڽǶȣ���ֱ����಻�ϼ�С���ɽ��ܷ�Χ
    % ps������ֻ���һ������������Ҳ������Զ�������𼶽��н��������0.2�����������һ����������һ�Σ�֮������С��0.005��������������һ����������
    r_num  =per_method_num-1;
    for iiqweqwe = 1:99
        %�տ�ʼһ����С���趨�����ģ������������������0.05������������
        if (abs(per_block_augnum-final_num)/per_block_augnum)>0.05
            r_num=r_num+1;
        end
        %�������������������������
        r_step   = (r_2-r_1)/(r_num);
        flag1 = 0;
        for angel = r_1:r_step:r_2
            flag1 = flag1+1;
        end
        final_num = flag1*flag2*flag3;
        %�жϣ����С��0.05��break����������趨����Ҳbreak
        if (abs(per_block_augnum-final_num)/per_block_augnum)<=0.05  || final_num>=per_block_augnum
            break;
        end 
    end
    disp(['final aug num is:',num2str(final_num*2)]);
    
    
    
    %%  aug
    aug_count_num = 0;
    image_roi_output = zeros(size(data));
    %ԭ����
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
                %��ת֮����Ϊ��ֵ���������1�����֣����������Ҫ��һ��
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
    %��ת=====================================================================================
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
                %��ת֮����Ϊ��ֵ���������1�����֣����������Ҫ��һ��
                %������һ��֮���Ƿ��Ӱ��֮ǰ��׼����Ч���أ���ʱ��֪��
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
% %��ת֮����Ϊ��ֵ���������1�����֣����������Ҫ��һ��
% figure;imshowpair(qwe,image_rotate_o, 'mon');title('4');
% max(max(image_rotate_o))









