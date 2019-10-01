%% ͳ�Ƹ��������Ϣ
% ��ʼ��
% clear;
% clc;
% close all;

%% load �ļ�
subject_log_path = '/data/@data_laowang/old/@@flow2/4aug_h5/subject';
wkspace = load(strcat(subject_log_path,filesep,'subject.mat'));
subject = wkspace.subject;

augdict.class_a_id = wkspace.augdict.class_a_id;% �ֶ�����a�ಡ�˵�id
augdict.class_b_id = wkspace.augdict.class_b_id;% �ֶ�����b�ಡ�˵�id

%% ��ȡ��Ϣ
id = {};
num_id = [];
tumor_size = [];
tumor_size_all = [];
voxel_size = {};
label = [];
for i = 1:length(subject)
   % ��ȡid
   id{end+1} =  num2str(subject(i).id);
   num_id(end+1) = subject(i).id;
   if ismember(subject(i).id, augdict.class_a_id)
       label(end+1) = 1;
   else
       label(end+1) = 2;
   end
   % ��ȡ�����������Դ�С
   tumor_size = autoadd(tumor_size,subject(i).tumor_size);
   % ��ȡ�����������Դ�С�ĺ�
   tumor_size_all(end+1) = subject(i).tumor_size_all; 
   % ��ȡ����ԭʼ�ռ��С��Ϣ
   voxel_size{end+1} =  subject(i).voxel_size; 
end

%% voxel �ߴ�
voxel_volume = [];%���ص�ʵ���������mm��
voxel_x = [];
voxel_z = [];

for i  = 1:length(subject)
    tmp_voxel_size = voxel_size{i};
    tmp_volume = [];
    tmp_x = [];
    tmp_z = [];
    for ii = 1:length(tmp_voxel_size)
        tmp_voxel = tmp_voxel_size{ii};
        tmpcolume = cumprod(tmp_voxel);% �������ص�ʵ���������mm��
        tmp_volume = [tmp_volume,tmpcolume(end)];
        tmp_x = [tmp_x,tmp_voxel(1)];
        tmp_z = [tmp_z,tmp_voxel(3)];
    end
    
    voxel_volume = autoadd(voxel_volume,tmp_volume);
    voxel_x = autoadd(voxel_x,tmp_x);
    voxel_z = autoadd(voxel_z,tmp_z);
   
end

%��״ͼ
figure;
subplot(2,1,1);% ���ص����ά�ȵĳߴ�
bar(voxel_x,'group','EdgeColor','y');% Ҳ���Ǻ����ķֱ���
xtb = get(gca,'XTickLabel');% ��ȡ��������ǩ���
xt = 1:length(subject);% ��ȡ�������̶Ⱦ��
yt = xt*0.01; % ��ȡ�������̶Ⱦ��
xtextp=xt-0.1;%ÿ����ǩ����λ�õĺ���꣬�����ȻӦ�ú�ԭ����һ���ˡ�                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('�����˵ĸ�CT�ĺ����ֱ��ʣ�mm��');
ylabel('�ֱ���(mm)');

subplot(2,1,2);% ���ص����ά�ȵĳߴ�
bar(voxel_z,'group','EdgeColor','y'); % Ҳ���ǲ��
xtb = get(gca,'XTickLabel');% ��ȡ��������ǩ���
xt = 1:length(subject);% ��ȡ�������̶Ⱦ��
yt = xt*0.01; % ��ȡ�������̶Ⱦ��
xtextp=xt-0.1;%ÿ����ǩ����λ�õĺ���꣬�����ȻӦ�ú�ԭ����һ���ˡ�                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('�����˵ĸ�CT�Ĳ��mm��');
ylabel('���(mm)');
%% �����˸���������ʵ���������cm������������
tumor_size = [];% ��������
tumor_voxel_volume = [];% ÿ������������ʵ�����
tumor_num = [];
for i  = 1:length(subject)
    tumor_size = autoadd(tumor_size,subject(i).tumor_size);
    tmp_tumor_num = [];
    voxel_size_per_tumor = [];
    for ii = 1:length(subject(i).v_m_id) % ����ÿһ��v
       v_tumor_num = length(subject(i).v_m_id{ii}) ; % �����ǰv���ж��ٸ�����
       tmp_tumor_num = [tmp_tumor_num,v_tumor_num];
       voxel_size_per_tumor = [voxel_size_per_tumor,ones(1,v_tumor_num)*voxel_volume(i,ii)];% ii����v�����
    end
    tumor_voxel_volume = autoadd(tumor_voxel_volume,voxel_size_per_tumor);
    tumor_num = autoadd(tumor_num,tmp_tumor_num);

end


tumor_volume = (tumor_size.*tumor_voxel_volume)/1e3;% ʵ�����(����cm)

%��״ͼ
figure;
subplot(3,1,1);% ���ص����ά�ȵĳߴ�
bar(tumor_size,'stack','EdgeColor','y');
xtb = get(gca,'XTickLabel');% ��ȡ��������ǩ���
xt = 1:length(subject);% ��ȡ�������̶Ⱦ��
yt = xt*0.01; % ��ȡ�������̶Ⱦ��
xtextp=xt;%ÿ����ǩ����λ�õĺ���꣬�����ȻӦ�ú�ԭ����һ���ˡ�                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('�����˵ĸ�CT�ĸ�������������');
ylabel('������');


subplot(3,1,2);% ���ص����ά�ȵĳߴ�
bar(tumor_volume,'stack','EdgeColor','y');
xtb = get(gca,'XTickLabel');% ��ȡ��������ǩ���
xt = 1:length(subject);% ��ȡ�������̶Ⱦ��
yt = xt*0.01; % ��ȡ�������̶Ⱦ��
xtextp=xt;%ÿ����ǩ����λ�õĺ���꣬�����ȻӦ�ú�ԭ����һ���ˡ�                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('�����˵ĸ�CT�ĸ���������ʵ�����');
ylabel('�������cm��');


subplot(3,1,3);% ���ص����ά�ȵĳߴ�
bar(tumor_num,'group','EdgeColor','y');
xtb = get(gca,'XTickLabel');% ��ȡ��������ǩ���
xt = 1:length(subject);% ��ȡ�������̶Ⱦ��
yt = xt*0.01; % ��ȡ�������̶Ⱦ��
xtextp=xt;%ÿ����ǩ����λ�õĺ���꣬�����ȻӦ�ú�ԭ����һ���ˡ�                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('�����˵ĸ�CT����������');
ylabel('��������');



%% ɢ��ͼ��ֱ��ͼ������������������һ��������������ֲ�
%�����������������
all_tumor_volume = tumor_volume(tumor_volume ~= 0);
figure;
subplot(3,1,1);
[n1,x1] = hist(all_tumor_volume);
h=bar(x1,n1,'hist');
set(h,'facecolor','r')
h = findobj(gca,'Type','patch');
h.EdgeColor = 'y';
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
title('�����������ֱ��ͼ');


%����һ���ֱ��ͼ
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
title(strcat('A���������������ֱ��ͼ,����',num2str(size(a_all_tumor_volume,1))));

subplot(3,1,3);
[n1,x1] = hist(b_all_tumor_volume);
h=bar(x1,n1,'hist');
set(h,'facecolor','r')
h = findobj(gca,'Type','patch');
h.EdgeColor = 'y';
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
title(strcat('B���������������ֱ��ͼ,����',num2str(size(b_all_tumor_volume,1))));

%% ����ǰ��block�������Աȣ��Լ����Աȣ������п���һ�������ж��ͼ�񣬶������������ɢ��ͼûʲô����ò�ƣ�ֻ��ֱ��ͼ�����ˣ�
% ��������������ڿ������xxx

adjust_voxelsize = wkspace.adjust_voxelsize;% �ز����ֱ���

or_block = [];%ÿ�����ÿ���������ֳ��˶��ٿ�

aug_block = [];%ÿ����һ���� �������ٿ�
aug_tumor_block = [];%ÿ�����ÿ������һ���� �����˶��ٿ�
aug_volume = [];% ÿ�����ÿ���������ձ� ���������



for i  = 1:length(subject)
    blocks_per_tumor = subject(i).blocks_num_per_tumor;
    blocks_per_tumor_aug = subject(i).per_tumor_aug_num;
    aug_block = autoadd(aug_block,subject(i).all_aug_num);

    tmp_blocks = [];
    tmp_blocks_aug = [];
    for ii = 1:length(subject(i).v_m_id) % ����ÿһ��v
       tmp_blocks = [tmp_blocks,blocks_per_tumor{ii}];
       tmp_blocks_aug = [tmp_blocks_aug,blocks_per_tumor_aug{ii}];
    end
    
    
    or_block = autoadd(or_block,tmp_blocks);
    aug_tumor_block =  autoadd(aug_tumor_block,tmp_blocks_aug);
        
    
end


tmpcolume = cumprod(adjust_voxelsize);% �������ص�ʵ���������mm��
tmp_voxel_num = cumprod(wkspace.augdict.savefomat.param);% ��������block�����ע�⣬�����ϸ�ģ���Ϊ�ֿ��ʱ����padding��
aug_volume = aug_tumor_block*tmpcolume(end)*tmp_voxel_num(end)/1e3;% ����cm



figure;
subplot(2,2,1);
bar(or_block,'stack','EdgeColor','y');
xtb = get(gca,'XTickLabel');% ��ȡ��������ǩ���
xt = 1:length(subject);% ��ȡ�������̶Ⱦ��
yt = xt*0.01; % ��ȡ�������̶Ⱦ��
xtextp=xt;%ÿ����ǩ����λ�õĺ���꣬�����ȻӦ�ú�ԭ����һ���ˡ�                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('�����˵ĸ��������ķֿ���');


subplot(2,2,2);
bar(aug_tumor_block,'stack','EdgeColor','y');
xtb = get(gca,'XTickLabel');% ��ȡ��������ǩ���
xt = 1:length(subject);% ��ȡ�������̶Ⱦ��
yt = xt*0.01; % ��ȡ�������̶Ⱦ��
xtextp=xt;%ÿ����ǩ����λ�õĺ���꣬�����ȻӦ�ú�ԭ����һ���ˡ�                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('�����˵ĸ�������������ĵķֿ���');


subplot(2,2,3);
bar(aug_block,'stack','EdgeColor','y');
xtb = get(gca,'XTickLabel');% ��ȡ��������ǩ���
xt = 1:length(subject);% ��ȡ�������̶Ⱦ��
yt = xt*0.01; % ��ȡ�������̶Ⱦ��
xtextp=xt;%ÿ����ǩ����λ�õĺ���꣬�����ȻӦ�ú�ԭ����һ���ˡ�                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('��������������ܿ���');

subplot(2,2,4);
bar(aug_volume,'stack','EdgeColor','y');
xtb = get(gca,'XTickLabel');% ��ȡ��������ǩ���
xt = 1:length(subject);% ��ȡ�������̶Ⱦ��
yt = xt*0.01; % ��ȡ�������̶Ⱦ��
xtextp=xt;%ÿ����ǩ����λ�õĺ���꣬�����ȻӦ�ú�ԭ����һ���ˡ�                    
ytextp=-0.6*yt(3)*ones(1,length(xt));
text(xtextp,ytextp,id,'HorizontalAlignment','right','rotation',90)
set(gca,'XTickLabel',[]);
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
grid on;
title('����������������������ܿ������');
ylabel('�������cm��');












