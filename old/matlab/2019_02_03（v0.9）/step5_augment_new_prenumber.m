clc;
clear;
mat_path = 'G:\$code_dasheng\final_data\@new_data\4block\bufufa';
mat_savepath = 'H:\@data_dasheng_zhuanyi_dl\@aug_data';
filename_list = dir(strcat(mat_path,filesep,'*.mat'));

subject_path = 'G:\$code_dasheng\final_data\@new_data\1mat\bufufa';%������Ҫ��Ӧ�������������1mat���ɣ���Ҫblock�ļ���
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
% buzhuanyi
label_1 = 0;label_2 = 1;label_3 = 0;
%% 
label = [label_1;label_2;label_3];
%% 
allnum=0;
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
               allnum = allnum+1
            end
        end
    end
    %��ת
    for angel = r_1:r_step:r_2
        for m = ad1_1:ad1_step:ad1_2
            for n = ad2_1:ad2_step:ad2_2
               allnum = allnum+1
            end
        end
    end
    
    
end
allnum

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









