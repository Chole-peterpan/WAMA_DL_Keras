%% 统计各个样本的信息
% 初始化
% clear;
% clc;
% close all;

%% load 文件
subject_log_path = 'H:\@data_NENs_response\EP\data\3Aug20000\subject';
wkspace = load(strcat(subject_log_path,filesep,'subject.mat'));
subject = wkspace.subject;


augdict.class_a_id = wkspace.augdict.class_a_id;% 手动传入a类病人的id
augdict.class_b_id = wkspace.augdict.class_b_id;% 手动传入b类病人的id

%% 提取信息
id = {};
num_id = [];
tumor_size = [];
tumor_size_all = [];
voxel_size = {};
label = [];
for i = 1:length(subject)
   % 提取id
   id{end+1} =  num2str(subject(i).id);
   num_id(end+1) = subject(i).id;
   if ismember(subject(i).id, augdict.class_a_id)
       label(end+1) = 1;
   else
       label(end+1) = 2;
   end
   % 提取各个肿瘤粗略大小
   tumor_size = autoadd(tumor_size,subject(i).tumor_size);
   % 提取所有肿瘤粗略大小的和
   tumor_size_all(end+1) = subject(i).tumor_size_all; 
   % 提取体素原始空间大小信息
   voxel_size{end+1} =  subject(i).voxel_size; 
end

%% voxel 尺寸
voxel_volume = [];%体素的实际体积（立方mm）
voxel_x = [];
voxel_z = [];

for i  = 1:length(subject)
    tmp_voxel_size = voxel_size{i};
    tmp_volume = [];
    tmp_x = [];
    tmp_z = [];
    for ii = 1:length(tmp_voxel_size)
        tmp_voxel = tmp_voxel_size{ii};
        tmpcolume = cumprod(tmp_voxel);% 计算体素的实际体积（立方mm）
        tmp_volume = [tmp_volume,tmpcolume(end)];
        tmp_x = [tmp_x,tmp_voxel(1)];
        tmp_z = [tmp_z,tmp_voxel(3)];
    end
    
    voxel_volume = autoadd(voxel_volume,tmp_volume);
    voxel_x = autoadd(voxel_x,tmp_x);
    voxel_z = autoadd(voxel_z,tmp_z);
   
end

%柱状图
figure;
subplot(2,1,1);% 体素的三个维度的尺寸
bar(voxel_x,'group','EdgeColor','y');% 也就是横截面的分辨率
xtb = get(gca,'XTickLabel');% 获取横坐标轴标签句柄
xt = 1:length(subject);% 获取横坐标轴刻度句柄
yt = xt*0.01; % 获取纵坐标轴刻度句柄
xtextp=xt-0.1;%每个标签放置位置的横坐标，这个自然应该和原来的一样了。                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('各病人的各CT的横截面分辨率（mm）');
ylabel('分辨率(mm)');

subplot(2,1,2);% 体素的三个维度的尺寸
bar(voxel_z,'group','EdgeColor','y'); % 也就是层厚
xtb = get(gca,'XTickLabel');% 获取横坐标轴标签句柄
xt = 1:length(subject);% 获取横坐标轴刻度句柄
yt = xt*0.01; % 获取纵坐标轴刻度句柄
xtextp=xt-0.1;%每个标签放置位置的横坐标，这个自然应该和原来的一样了。                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('各病人的各CT的层厚（mm）');
ylabel('层厚(mm)');
%% 各病人各个肿瘤的实际体积（立方cm）与像素数量
tumor_size = [];% 体素数量
tumor_voxel_volume = [];% 每个肿瘤的体素实际体积
tumor_num = [];
for i  = 1:length(subject)
    tumor_size = autoadd(tumor_size,subject(i).tumor_size);
    tmp_tumor_num = [];
    voxel_size_per_tumor = [];
    for ii = 1:length(subject(i).v_m_id) % 遍历每一个v
       v_tumor_num = length(subject(i).v_m_id{ii}) ; % 求出当前v共有多少个肿瘤
       tmp_tumor_num = [tmp_tumor_num,v_tumor_num];
       voxel_size_per_tumor = [voxel_size_per_tumor,ones(1,v_tumor_num)*voxel_volume(i,ii)];% ii就是v的序号
    end
    tumor_voxel_volume = autoadd(tumor_voxel_volume,voxel_size_per_tumor);
    tumor_num = autoadd(tumor_num,tmp_tumor_num);

end


tumor_volume = (tumor_size.*tumor_voxel_volume)/1e3;% 实际体积(立方cm)

%柱状图
figure;
subplot(3,1,1);% 体素的三个维度的尺寸
bar(tumor_size,'stack','EdgeColor','y');
xtb = get(gca,'XTickLabel');% 获取横坐标轴标签句柄
xt = 1:length(subject);% 获取横坐标轴刻度句柄
yt = xt*0.01; % 获取纵坐标轴刻度句柄
xtextp=xt;%每个标签放置位置的横坐标，这个自然应该和原来的一样了。                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('各病人的各CT的各肿瘤体素数量');
ylabel('体素数');


subplot(3,1,2);% 体素的三个维度的尺寸
bar(tumor_volume,'stack','EdgeColor','y');
xtb = get(gca,'XTickLabel');% 获取横坐标轴标签句柄
xt = 1:length(subject);% 获取横坐标轴刻度句柄
yt = xt*0.01; % 获取纵坐标轴刻度句柄
xtextp=xt;%每个标签放置位置的横坐标，这个自然应该和原来的一样了。                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('各病人的各CT的各个肿瘤的实际体积');
ylabel('体积（立方cm）');


subplot(3,1,3);% 体素的三个维度的尺寸
bar(tumor_num,'group','EdgeColor','y');
xtb = get(gca,'XTickLabel');% 获取横坐标轴标签句柄
xt = 1:length(subject);% 获取横坐标轴刻度句柄
yt = xt*0.01; % 获取纵坐标轴刻度句柄
xtextp=xt;%每个标签放置位置的横坐标，这个自然应该和原来的一样了。                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('各病人的各CT的肿瘤个数');
ylabel('肿瘤个数');



%% 散点图或直方图，把所有肿瘤，单独一类的肿瘤都看看分布
%所有肿瘤的体积，乱序
all_tumor_volume = tumor_volume(tumor_volume ~= 0);
figure;
subplot(3,1,1);
[n1,x1] = hist(all_tumor_volume);
h=bar(x1,n1,'hist');
set(h,'facecolor','r')
h = findobj(gca,'Type','patch');
h.EdgeColor = 'y';
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
title('所有肿瘤体积直方图');


%单独一类的直方图
class_a_tumor_volume = tumor_volume(label==1,:);
class_b_tumor_volume = tumor_volume(label==2,:);
a_all_tumor_volume = class_a_tumor_volume(class_a_tumor_volume ~= 0);
b_all_tumor_volume = class_b_tumor_volume(class_b_tumor_volume ~= 0);

subplot(3,1,2);
[n1,x1] = hist(a_all_tumor_volume);
h=bar(x1,n1,'hist');
set(h,'facecolor','r')
h = findobj(gca,'Type','patch');
h.EdgeColor = 'y';
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
title(strcat('A类中所有肿瘤体积直方图,样本数：',num2str(size(a_all_tumor_volume,1))));

subplot(3,1,3);
[n1,x1] = hist(b_all_tumor_volume);
h=bar(x1,n1,'hist');
set(h,'facecolor','r')
h = findobj(gca,'Type','patch');
h.EdgeColor = 'y';
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
title(strcat('B类中所有肿瘤体积直方图,样本数：',num2str(size(b_all_tumor_volume,1))));

%% 扩增前后block的数量对比，以及体积对比（不过有可能一个病人有多个图像，多个肿瘤，所以散点图没什么意义貌似，只做直方图就行了）
% 扩增后的总体积等于块数乘以xxx

adjust_voxelsize = wkspace.adjust_voxelsize;% 重采样后分辨率

or_block = [];%每个样本的每个肿瘤被分成了多少块

aug_block = [];%每个样本一共被 扩增多少块
aug_tumor_block = [];%每个样本的每个肿瘤一共被 扩增了多少块
aug_volume = [];% 每个样本的每个肿瘤最终被 扩增的体积



for i  = 1:length(subject)
    blocks_per_tumor = subject(i).blocks_num_per_tumor;
    blocks_per_tumor_aug = subject(i).per_tumor_aug_num;
    aug_block = autoadd(aug_block,subject(i).all_aug_num);

    tmp_blocks = [];
    tmp_blocks_aug = [];
    for ii = 1:length(subject(i).v_m_id) % 遍历每一个v
       tmp_blocks = [tmp_blocks,blocks_per_tumor{ii}];
       tmp_blocks_aug = [tmp_blocks_aug,blocks_per_tumor_aug{ii}];
    end
    
    
    or_block = autoadd(or_block,tmp_blocks);
    aug_tumor_block =  autoadd(aug_tumor_block,tmp_blocks_aug);
        
    
end


tmpcolume = cumprod(adjust_voxelsize);% 计算体素的实际体积（立方mm）
tmp_voxel_num = cumprod(wkspace.augdict.savefomat.param);% 单个扩增block的体积（注意，不是严格的，因为分块的时候有padding）
aug_volume = aug_tumor_block*tmpcolume(end)*tmp_voxel_num(end)/1e3;% 立方cm



figure;
subplot(2,2,1);
bar(or_block,'stack','EdgeColor','y');
xtb = get(gca,'XTickLabel');% 获取横坐标轴标签句柄
xt = 1:length(subject);% 获取横坐标轴刻度句柄
yt = xt*0.01; % 获取纵坐标轴刻度句柄
xtextp=xt;%每个标签放置位置的横坐标，这个自然应该和原来的一样了。                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('各病人的各个肿瘤的分块数');


subplot(2,2,2);
bar(aug_tumor_block,'stack','EdgeColor','y');
xtb = get(gca,'XTickLabel');% 获取横坐标轴标签句柄
xt = 1:length(subject);% 获取横坐标轴刻度句柄
yt = xt*0.01; % 获取纵坐标轴刻度句柄
xtextp=xt;%每个标签放置位置的横坐标，这个自然应该和原来的一样了。                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('各病人的各个肿瘤扩增后的的分块数');


subplot(2,2,3);
bar(aug_block,'stack','EdgeColor','y');
xtb = get(gca,'XTickLabel');% 获取横坐标轴标签句柄
xt = 1:length(subject);% 获取横坐标轴刻度句柄
yt = xt*0.01; % 获取纵坐标轴刻度句柄
xtextp=xt;%每个标签放置位置的横坐标，这个自然应该和原来的一样了。                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('各病人扩增后的总块数');

subplot(2,2,4);
bar(aug_volume,'stack','EdgeColor','y');
xtb = get(gca,'XTickLabel');% 获取横坐标轴标签句柄
xt = 1:length(subject);% 获取横坐标轴刻度句柄
yt = xt*0.01; % 获取纵坐标轴刻度句柄
xtextp=xt;%每个标签放置位置的横坐标，这个自然应该和原来的一样了。                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('各病人扩增后各个肿瘤的总块数体积');
ylabel('体积（立方cm）');












