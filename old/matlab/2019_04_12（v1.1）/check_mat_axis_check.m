clc;
clear;
mat_path = 'G:\@data_zhuanyi_zhongshan\PNENs_wang\3cut_keepsize';
% mat_path = 'G:\@data_zhuanyi_zhongshan\PNENs_wang\3cut';
nii_filename_list = dir(strcat(mat_path,filesep,'*.mat'));

%% eye view
figure;
for ii = 1:length(nii_filename_list)
    filename = nii_filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    data = workspaces.Data_cu;
    mask = workspaces.mask_cu;
    for iii=1:size(data,3)
        imshowpair(data(:,:,iii), mask(:,:,iii), 'falsecolor');
        title(filename);
        pause(0.05);
    end
end


%% volume histogram（体积是粗略计算的，毕竟roi就是粗略的）
value_mat = zeros(length(nii_filename_list),3);
for ii = 1:length(nii_filename_list)
    filename = nii_filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    value_mat(ii,:) = workspaces.or_size;
end
value_mat(:,4)=value_mat(:,1).*value_mat(:,2).*value_mat(:,3);
for ii = 1:4
value_mat(:,ii)=sort(value_mat(:,ii));
end


figure;
subplot(2,2,1);plot(value_mat(:,1));title(strcat('x :max is :',num2str(max(value_mat(:,1))),'; min is :',num2str(min(value_mat(:,1)))));
subplot(2,2,2);plot(value_mat(:,2));title(strcat('y :max is :',num2str(max(value_mat(:,2))),'; min is :',num2str(min(value_mat(:,2)))));
subplot(2,2,3);plot(value_mat(:,3));title(strcat('thickness :max is :',num2str(max(value_mat(:,3))),'; min is :',num2str(min(value_mat(:,3)))));
subplot(2,2,4);plot(value_mat(:,4));title(strcat('volume :max is :',num2str(max(value_mat(:,4))),'; min is :',num2str(min(value_mat(:,4)))));






