%% �������ݣ����г�����������б��Թ����������趨��
var_flag = ~test_flag;%��֤���Ͳ��Լ�ֻ�ܶ�ѡһ�۲�

if test_flag
    tmp_str = '@test';
else
    tmp_str = '@ver';
end

% ���ֿ������
per_block_name_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_@_name_per_block.txt'];
% ���ֿ��loss
per_block_loss_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_@_loss_per_block.txt'];
% ���ֿ��label
per_block_label_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_@_label_per_block.txt'];
% ���ֿ��Ԥ��ֵ
per_block_pre_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_@_pre_per_block.txt'];

% �������˵�loss���ɸò�����������������blockƽ���õ���
peron_loss_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_loss_per_person.txt'];
% �������˵�id
peron_id_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_id_per_person.txt'];
% �������˵�label
peron_label_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_label_per_person.txt'];
% �������˵�pre���ɸò�����������������block��ֵ+�ж�+ȡ��ֵ�õ���
peron_pre_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_pre_per_person.txt'];

% ���в���ƽ�����loss
loss_path = [logpath,filesep,'@',num2str(foldname),'_loss_',tmp_str(2:end),'.txt'];


% ��������
per_block_loss = importdata(per_block_loss_path);
per_block_name = importdata(per_block_name_path);
per_block_label = importdata(per_block_label_path);
per_block_pre = importdata(per_block_pre_path);

person_loss = importdata(peron_loss_path);
person_id = importdata(peron_id_path);
person_label = importdata(peron_label_path);
person_pre = importdata(peron_pre_path);

loss_all = importdata(loss_path);


%% person_id Ԥ����
% �������޸�bug ----------------------------
person_id = person_id{1};
person_id = split(person_id,'@');
person_id = person_id{2};
person_id = person_id(2:end-1);
person_id = split(person_id,',');
% -----------------------------------------
% person_id = person_id{1};
% person_id = person_id(4:end-1);
% person_id = split(person_id,',');
% ���ַ���ǰ��Ŀո�ȥ��
for i = 2:length(person_id)
   person_id{i} = person_id{i}(2:end); 
end

%% per_block_name Ԥ�������Ϊname_mat
block_name = per_block_name{1};
block_name = split(block_name,'@');
block_name = block_name{2};
block_name = block_name(2:end-1);
% block_name = block_name(4:end-1);
block_name = split(block_name,',');
per_block_name = block_name;% Ԥ����
% ���ַ���ǰ��Ŀո�ȥ��
for i = 2:length(block_name)
   block_name{i} = block_name{i}(2:end); 
end
% ���ַ���ת��Ϊn*4��mat��ÿһ�зֱ��Ӧsvmb
name_mat = [];
for i = 1:length(block_name)
    tmp_str = block_name{i}(2:end-1);
    tmp_vector = split(tmp_str,'_');
    tmp_mat = [];
    for ii = 1:length(tmp_vector)
    tmp_mat = [tmp_mat,str2double(tmp_vector{ii}(2:end))];
    end
    name_mat = [name_mat;tmp_mat];
end
block_name = name_mat;
clear name_mat;


%% ��Ҫ���ݵ�Ԥ����
%% 1 ��ȡ�ܵ�loss
all_loss = [];
all_loss_iter = [];
for ii = 1:length(loss_all)
    tmp_str = loss_all{ii};
    index = find(tmp_str=='@');
    all_loss(ii) = str2double(tmp_str(index+1:end));
    all_loss_iter(ii) = str2double(tmp_str(1:index-1));
end

%% 2 ��ȡ��Ϊ��λ�ģ�person_pre Ԥ����
person_pre_all = zeros(length(person_id),length(person_pre));
person_pre_iter = zeros(length(person_pre),1);
for ii = 1:length(person_pre)
    tmp_str = person_pre{ii};
    tmp_str = split(tmp_str,'@');
    person_pre_iter(ii) = str2double(tmp_str{1});
    
    tmp_str = tmp_str{2}(2:end-1);
    tmp_str = split(tmp_str,',');
    tmp_loss = [];
    for iii =  1:length(tmp_str)
        tmp_loss(end+1) = str2double(tmp_str{iii});
    end
    tmp_loss = tmp_loss';
    person_pre_all(:,ii) = tmp_loss;
end
person_pre_iter =person_pre_iter';
clear person_pre

%% 3 ��ȡ��Ϊ��λ�ģ�person_loss Ԥ����iterֱ����ǰ��person_pre�ģ�
person_loss_all = zeros(length(person_id),length(person_loss));
for ii = 1:length(person_loss)
    tmp_str = person_loss{ii};
    tmp_str = split(tmp_str,'@');
    
    tmp_str = tmp_str{2}(2:end-1);
    tmp_str = split(tmp_str,',');
    tmp_loss = [];
    for iii =  1:length(tmp_str)
        tmp_loss(end+1) = str2double(tmp_str{iii});
    end
    tmp_loss = tmp_loss';
    person_loss_all(:,ii) = tmp_loss;
end
clear person_loss

%% 4 ��ȡblockΪ��λ�ģ�block����preԤ����iterֱ����ǰ��person_pre�ģ�
block_pre = zeros(length(per_block_name),length(per_block_pre));
for ii = 1: length(per_block_pre)
    tmp_str = per_block_pre{ii};
    tmp_str = split(tmp_str,'@');    
    tmp_str = tmp_str{2}(2:end-1);
    tmp_str = split(tmp_str,',');
    tmp_pre = [];
    for iii =  1:length(tmp_str)
        tmp_pre(end+1) = str2double(tmp_str{iii});
    end
    tmp_pre = tmp_pre';
    block_pre(:,ii) = tmp_pre;
end
clear per_block_pre

%% 5 ��ȡblockΪ��λ�ģ�block����lossԤ����iterֱ����ǰ��person_pre�ģ�
block_loss = zeros(length(per_block_name),length(per_block_loss));
for ii = 1: length(per_block_loss)
    tmp_str = per_block_loss{ii};
    tmp_str = split(tmp_str,'@');    
    tmp_str = tmp_str{2}(2:end-1);
    tmp_str = split(tmp_str,',');
    tmp_loss = [];
    for iii =  1:length(tmp_str)
        tmp_loss(end+1) = str2double(tmp_str{iii});
    end
    tmp_loss = tmp_loss';
    block_loss(:,ii) = tmp_loss;
end
clear per_block_loss


%% 6 ��ȡperson��pre label��true label
person_pre_label = double(person_pre_all >= acc_thresold);
person_true_label = [];
person_label = person_label{1};
person_label = split(person_label,'@');
person_label = person_label{2};
person_label = person_label(2:end-1);
% person_label = person_label(4:end-1);
person_label = split(person_label,',');
% ���ַ���ǰ��Ŀո�ȥ��
for i = 1:length(person_label)
   person_true_label(i) = str2double(person_label{i}); 
end
person_true_label = person_true_label';
tmp_label = [];
for i = 1:length(person_pre_iter)
    tmp_label = [tmp_label,person_true_label];
end
person_true_label = tmp_label;clear tmp_label;






%% 7 ��ȡblock��pre label��true label
block_pre_label = double(block_pre>=acc_thresold);
block_true_label = [];
block_label = per_block_label{1};
block_label = split(block_label,'@');
block_label = block_label{2};
block_label = block_label(2:end-1);
% block_label = block_label(4:end-1);
block_label = split(block_label,',');
% ���ַ���ǰ��Ŀո�ȥ��
for i = 2:length(block_label)
   block_true_label(i) = str2double(block_label{i}); 
end
block_true_label = block_true_label';
tmp_label = [];
for i = 1:length(person_pre_iter)
    tmp_label = [tmp_label,block_true_label];
end
block_true_label = tmp_label;clear tmp_label;












