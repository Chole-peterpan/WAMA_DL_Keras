%��ʼ��
clc;
clear;
%��������
block_mat_path = 'G:\@data_NENs_recurrence\PNENs\data\a\2block';%block
mat_savepath = 'H:\@data_NENs_recurrent\3aug_35000';
aug_num = 35000;%����������
a_b_ratio = [1,1];%����AB����Ϊ1:2��������Ϊ[1,2],A�Ƿ�PD��B��PD
 
%==========================================================================
aug_num_A = aug_num*(a_b_ratio(1)/sum(a_b_ratio));%A������������������
aug_num_B = aug_num*(a_b_ratio(2)/sum(a_b_ratio));%B������������������
%���ز�����Ϣ
workspaces = load(strcat(block_mat_path,filesep,'subject_all',filesep,'subject.mat'));
subject_all = workspaces.subject_all;%��¼����ͳ����Ϣ��mat�ļ���Ϊ�ṹ�����顣
%�����˼��ϱ�ǩ=======================================================================
for i = 1:length(subject_all)
    %���������������򣬸���������Ӧ��label
    %����ǰ20�����������������Ǹ�������ʵ������
    if subject_all(i).id <= 49 %��PD
        label_1 = 0;label_2 = 0;label_3 = 1;%�ֶ�����
    else %PD
        label_1 = 0;label_2 = 1;label_3 = 0;%�ֶ�����
    end
    subject_all(i).label = [label_1;label_2;label_3];
    subject_all(i).label1 = [label_1;label_2];
    subject_all(i).label2 = [label_1;label_3];
    subject_all(i).label3 = [label_2;label_3];
    
    subject_all(i).label_1 = label_1;
    subject_all(i).label_2 = label_2;
    subject_all(i).label_3 = label_3;
end


%% ��ʼ����wdnmd
sub_num_A = 0;%��PD����
sub_num_B = 0;%PD����
for i = 1:length(subject_all)
    if subject_all(i).label(3) == 1 %��PD
        sub_num_A = sub_num_A+1;
    elseif  subject_all(i).label(2) == 1 %PD
        sub_num_B = sub_num_B+1;
    end
end

per_sub_augnum_A = floor(aug_num_A/sub_num_A);%��PD ÿ�����������Ĵ������
per_sub_augnum_B = floor(aug_num_B/sub_num_B);%PD ÿ�����������Ĵ������



for i = 1:length(subject_all)
    subject = subject_all(i);
    blocks = subject.blocks_num_all;
    % �ж����
    if subject.label(3) == 1 %��PD
        per_sub_augnum = per_sub_augnum_A;
    elseif  subject.label(2) == 1 %PD
        per_sub_augnum = per_sub_augnum_B;
    end
    
    per_block_augnum = (per_sub_augnum/blocks)/2;
    %per_block_augnumΪÿ������Ҫ����������,������Ҫע�⣬��Ϊ�з�ת�����������Ҫ�ٳ���2
    %�ٸ�������Ŀ�����������ȷ�������Ĳ�������ת�Ƕȷ�Χ��������
    %����������Χ��Ҳ��������������ˣ��趨ÿ��������������������ͬ������Ҫ�����η� 
    per_method_num = floor(nthroot(per_block_augnum,3)); %����ÿ���������������Ĵ������趨ÿ�ַ��������Ĵ�����ͬ��
    %ad�����������ǶԱȶȵ��ڵĲ�����lowin��highin����Ϊͼ���Ѿ���һ���ˣ�����lowin��0��0.1�м�ȡ��highin��0.8��1֮��ȡ��
    ad2_1 = 0.800; ad2_2 = 1.00;  ad2_step = abs((ad2_2-ad2_1)/(per_method_num-1));%��Ҫ-1����Ȼ��ʹ������������               
    ad1_1 = 0;     ad1_2 = 0.1;   ad1_step = abs((ad1_2-ad1_1)/(per_method_num-1));         
    r_1   = 0;     r_2   = 40;    r_step   = abs((r_2-r_1)/(per_method_num-1));  
    %����һ��Ҫ�Ӿ���ֵ����Ȼ��������С��ʱ�򣬻���ָ������������û�����ᵼ����������Ϊ0
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
    disp(['final block aug num is:',num2str(final_num*2)]);
    
    
    %��block��Ϣ���浽���˽ṹ��������
    subject_all(i).per_block_aug_num = final_num*2;
    subject_all(i).per_tumor_aug_num = final_num*2*(subject_all(i).blocks_num_per_tumor);
    subject_all(i).all_aug_num = sum(subject_all(i).per_tumor_aug_num);
    
    
    %%  aug �ڲ��˵�λ��,��ÿ��blockΪ��λ��������
    aug_count_num = 0;
    data_path=strcat(block_mat_path,filesep,num2str(subject_all(1).id),'_',num2str(1),'.mat');
    workspaces = load(data_path);
    data = workspaces.block;
    image_roi_output = zeros(size(data));
    
    for ii = 1:subject_all(i).blocks_num_all
        data_path=strcat(block_mat_path,filesep,num2str(subject_all(i).id),'_',num2str(ii),'.mat');
        workspaces = load(data_path);
        data = workspaces.block;
        data = mat2gray(data);%һ��Ҫ��һ������Ȼimadjust��ʱ���ضϺܴ�һ����ֵ������������������
        %ԭ����
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
                    %��ת֮����Ϊ��ֵ���������1�����֣����������Ҫ��һ��,����֮ǰ�Ѿ���׼���ˣ�����Ҳ�ɲ��ù�һ����
                    image_roi_output = mat2gray(image_roi_output);%һ��Ҫ��һ��,test����ҲҪ��һ��
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
        %��ת=====================================================================================
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
                    %��ת֮����Ϊ��ֵ���������1�����֣����������Ҫ��һ��
                    %������һ��֮���Ƿ��Ӱ��֮ǰ��׼����Ч���أ���ʱ��֪��
                    image_roi_output = mat2gray(image_roi_output);%һ��Ҫ��һ��,test����ҲҪ��һ��
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




