%������step5_augment.m����ļ�����ǰ�������������������º�������

clc;
clear;
mat_path = 'G:\@data_zhuanyi_zhongshan\PNENs_wang\4block\buzhuanyi';
filename_list = dir(strcat(mat_path,filesep,'*.mat'));


allnum=0;
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
    
%     %% ���ݿ�����ȷ�������Ķ���(��ʢ�ĸΰ�����ʵ�飩
%     if  blocks<5
%         a = 2; b= 2;  d = 1.2;      %  ����: 588    ������: 1386
%     elseif blocks<14
%         a = 4; b= 4;  d = 2.4;      %  ����: 84     ������: 198
%     else
%         a = 8; b= 5; d = 3.6;      %  ����: 30     ������: 63
%     end
    
    
    %% ���ݿ�����ȷ�������Ķ���(pnensת�Ƹ�������ɽ��
    if  blocks<5
        a = 2; b= 2;  d = 1.2;      %  ����: 588    ������: 1386
    elseif blocks<14
        a = 4; b= 4;  d = 2.4;      %  ����: 84     ������: 198
    else
        a = 8; b= 5; d = 3.6;      %  ����: 30     ������: 63
    end
    %% �ٸ��������ָ�꣬ȷ�������Ĳ�������ת�Ƕȷ�Χ��������
    %ad�������ǶԱȶȵ��ڵĲ�����lowin��highin����Ϊͼ���Ѿ���һ���ˣ�����lowin��0��0.1�м�ȡ��highin��0.8��1֮��ȡ��
    %��ת��
    ad2_1 = 0.800; ad2_step = 0.02*a;    ad2_2 = 1.00;             %����:0.015 ��ת�� :0.015
    ad1_1 = 0;     ad1_step = 0.02*b;    ad1_2 = 0.1;              %����:0.01  ��ת�� :0.01
    r_1   = 0;     r_step   = 4.8*d;     r_2   = 40;               %����:3.6   ��ת�� :3.6
    %ת��
    ad2_1 = 0.800; ad2_step = 0.02*a;    ad2_2 = 1.00;             %����:0.015 ��ת�� :0.015
    ad1_1 = 0;     ad1_step = 0.02*b;    ad1_2 = 0.1;              %����:0.01  ��ת�� :0.01
    r_1   = 0;     r_step   = 4.8*d;     r_2   = 40;               %����:3.6   ��ת�� :3.6
    
    
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









