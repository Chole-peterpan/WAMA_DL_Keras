%�����º������ᣬ�̶�����

clc;
clear;
mat_path = 'G:\code_dasheng\final_data\@new_data\4block\1fufa';
mat_savepath = 'F:\@data_pnens_zhuanyi_dl\aug';
filename_list = dir(strcat(mat_path,filesep,'*.mat'));


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
% zhuanyi
label_1 = 0;label_2 = 0;label_3 = 1;
% % bu zhuanyi
% label_1 = 0;label_2 = 1;label_3 = 0;




%% 
label = [label_1;label_2;label_3];

for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    data = workspaces.Data;
    %mask = workspaces.Mask; %û���õ�
    blocks = workspaces.blocks;
    
%     %% ���ݿ�����ȷ�������Ķ���(NENs�ּ�ʵ�飩
%     if  blocks<11
%         a = 6; b= 4;  d = 1.6;      %  G3: 72     G1G2: 36
%     elseif blocks<25
%         a = 7; b= 5;  d = 2.4;      %  G3: 36     G1G2: 18
%     else
%         a = 11; b= 5; d = 3.6;      %  G3: 18     G1G2: 12
%     end
    
    %% ���ݿ�����ȷ�������Ķ���(��ʢ�ĸΰ�����ʵ�飩
    if  blocks<5
        a = 2; b= 2;  d = 1.2;      %  ����: 588    ������: 1386
    elseif blocks<14
        a = 4; b= 4;  d = 2.4;      %  ����: 84     ������: 198
    else
        a = 8; b= 5; d = 3.6;      %  ����: 30     ������: 63
    end
    
    %% �ٸ��������ָ�꣬ȷ�������Ĳ�������ת�Ƕȷ�Χ��������
    %ad�������ǶԱȶȵ��ڵĲ�����lowin��highin����Ϊͼ���Ѿ���һ���ˣ�����lowin��0��0.1�м�ȡ��highin��0.8��1֮��ȡ��
    ad2_1 = 0.800; ad2_step = 0.015*a;    ad2_2 = 1.00;             %����:0.015 δ���� :0.015
    ad1_1 = 0;     ad1_step = 0.01*b;    ad1_2 = 0.1;              %����:0.01  δ���� :0.01
    r_1   = 0;     r_step   = 3.6*d;     r_2   = 40;               %����:3.6   δ���� :3.6
    
    %���������������
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
    disp(['each round aug num is:',num2str(final_num)]);
    
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
    %��ת
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









