%适用于step5_augment.m这个文件，提前计算扩增数量，可用新函数代替

clc;
clear;
mat_path = 'G:\@data_zhuanyi_zhongshan\PNENs_wang\4block\buzhuanyi';
filename_list = dir(strcat(mat_path,filesep,'*.mat'));


allnum=0;
for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    data = workspaces.Data;
    %mask = workspaces.Mask; %没有用到
    blocks = workspaces.blocks;
    
%     %% 根据块数量确定扩增的多少(NENs分级实验）
%     if  blocks<11
%         a = 6; b= 4;  d = 1.6;      %  G3: 72     G1G2: 36
%     elseif blocks<25
%         a = 7; b= 5;  d = 2.4;      %  G3: 36     G1G2: 18
%     else
%         a = 11; b= 5; d = 3.6;      %  G3: 18     G1G2: 12
%     end
    
%     %% 根据块数量确定扩增的多少(大盛的肝癌复发实验）
%     if  blocks<5
%         a = 2; b= 2;  d = 1.2;      %  复发: 588    不复发: 1386
%     elseif blocks<14
%         a = 4; b= 4;  d = 2.4;      %  复发: 84     不复发: 198
%     else
%         a = 8; b= 5; d = 3.6;      %  复发: 30     不复发: 63
%     end
    
    
    %% 根据块数量确定扩增的多少(pnens转移复发，中山）
    if  blocks<5
        a = 2; b= 2;  d = 1.2;      %  复发: 588    不复发: 1386
    elseif blocks<14
        a = 4; b= 4;  d = 2.4;      %  复发: 84     不复发: 198
    else
        a = 8; b= 5; d = 3.6;      %  复发: 30     不复发: 63
    end
    %% 再根据上面的指标，确定扩增的参数，旋转角度范围及步长，
    %ad那两个是对比度调节的参数（lowin和highin，因为图像已经归一化了，所以lowin在0到0.1中间取，highin在0.8到1之间取）
    %不转移
    ad2_1 = 0.800; ad2_step = 0.02*a;    ad2_2 = 1.00;             %复发:0.015 不转移 :0.015
    ad1_1 = 0;     ad1_step = 0.02*b;    ad1_2 = 0.1;              %复发:0.01  不转移 :0.01
    r_1   = 0;     r_step   = 4.8*d;     r_2   = 40;               %复发:3.6   不转移 :3.6
    %转移
    ad2_1 = 0.800; ad2_step = 0.02*a;    ad2_2 = 1.00;             %复发:0.015 不转移 :0.015
    ad1_1 = 0;     ad1_step = 0.02*b;    ad1_2 = 0.1;              %复发:0.01  不转移 :0.01
    r_1   = 0;     r_step   = 4.8*d;     r_2   = 40;               %复发:3.6   不转移 :3.6
    
    
    %计算最后扩增数量
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
    %原方向
    for angel = r_1:r_step:r_2
        for m = ad1_1:ad1_step:ad1_2
            for n = ad2_1:ad2_step:ad2_2
               allnum = allnum+1
            end
        end
    end
    %翻转
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
% %旋转之后因为插值会产生大于1的数字，所以最后需要归一化
% figure;imshowpair(qwe,image_rotate_o, 'mon');title('4');
% max(max(image_rotate_o))









