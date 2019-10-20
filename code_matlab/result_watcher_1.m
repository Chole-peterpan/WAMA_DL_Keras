%% 加载数据，并列出各个级别的列表，以供后续参数设定。
var_flag = ~test_flag;%验证集和测试集只能二选一观察

if test_flag
    tmp_str = '@test';
else
    tmp_str = '@ver';
end

% 各分块的名字
per_block_name_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_@_name_per_block.txt'];
% 各分块的loss
per_block_loss_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_@_loss_per_block.txt'];
% 各分块的label
per_block_label_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_@_label_per_block.txt'];
% 各分块的预测值
per_block_pre_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_@_pre_per_block.txt'];

% 各个病人的loss（由该病人所有肿瘤的所有block平均得到）
peron_loss_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_loss_per_person.txt'];
% 各个病人的id
peron_id_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_id_per_person.txt'];
% 各个病人的label
peron_label_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_label_per_person.txt'];
% 各个病人的pre（由该病人所有肿瘤的所有block均值+判断+取最值得到）
peron_pre_path = [logpath,filesep,tmp_str,filesep,'@',num2str(foldname),'_pre_per_person.txt'];

% 所有病人平均后的loss
loss_path = [logpath,filesep,'@',num2str(foldname),'_loss_',tmp_str(2:end),'.txt'];


% 加载数据
per_block_loss = importdata(per_block_loss_path);
per_block_name = importdata(per_block_name_path);
per_block_label = importdata(per_block_label_path);
per_block_pre = importdata(per_block_pre_path);

person_loss = importdata(peron_loss_path);
person_id = importdata(peron_id_path);
person_label = importdata(peron_label_path);
person_pre = importdata(peron_pre_path);

loss_all = importdata(loss_path);


%% person_id 预处理
% 新增，修复bug ----------------------------
person_id = person_id{1};
person_id = split(person_id,'@');
person_id = person_id{2};
person_id = person_id(2:end-1);
person_id = split(person_id,',');
% -----------------------------------------
% person_id = person_id{1};
% person_id = person_id(4:end-1);
% person_id = split(person_id,',');
% 把字符串前面的空格去掉
for i = 2:length(person_id)
   person_id{i} = person_id{i}(2:end); 
end

%% per_block_name 预处理，输出为name_mat
block_name = per_block_name{1};
block_name = split(block_name,'@');
block_name = block_name{2};
block_name = block_name(2:end-1);
% block_name = block_name(4:end-1);
block_name = split(block_name,',');
per_block_name = block_name;% 预处理
% 把字符串前面的空格去掉
for i = 2:length(block_name)
   block_name{i} = block_name{i}(2:end); 
end
% 把字符串转换为n*4的mat，每一列分别对应svmb
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


%% 重要数据的预处理
%% 1 获取总的loss
all_loss = [];
all_loss_iter = [];
for ii = 1:length(loss_all)
    tmp_str = loss_all{ii};
    index = find(tmp_str=='@');
    all_loss(ii) = str2double(tmp_str(index+1:end));
    all_loss_iter(ii) = str2double(tmp_str(1:index-1));
end

%% 2 获取人为单位的：person_pre 预处理
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

%% 3 获取人为单位的：person_loss 预处理（iter直接用前面person_pre的）
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

%% 4 获取block为单位的：block——pre预处理（iter直接用前面person_pre的）
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

%% 5 获取block为单位的：block——loss预处理（iter直接用前面person_pre的）
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


%% 6 获取person的pre label和true label
person_pre_label = double(person_pre_all >= acc_thresold);
person_true_label = [];
person_label = person_label{1};
person_label = split(person_label,'@');
person_label = person_label{2};
person_label = person_label(2:end-1);
% person_label = person_label(4:end-1);
person_label = split(person_label,',');
% 把字符串前面的空格去掉
for i = 1:length(person_label)
   person_true_label(i) = str2double(person_label{i}); 
end
person_true_label = person_true_label';
tmp_label = [];
for i = 1:length(person_pre_iter)
    tmp_label = [tmp_label,person_true_label];
end
person_true_label = tmp_label;clear tmp_label;






%% 7 获取block的pre label和true label
block_pre_label = double(block_pre>=acc_thresold);
block_true_label = [];
block_label = per_block_label{1};
block_label = split(block_label,'@');
block_label = block_label{2};
block_label = block_label(2:end-1);
% block_label = block_label(4:end-1);
block_label = split(block_label,',');
% 把字符串前面的空格去掉
for i = 2:length(block_label)
   block_true_label(i) = str2double(block_label{i}); 
end
block_true_label = block_true_label';
tmp_label = [];
for i = 1:length(person_pre_iter)
    tmp_label = [tmp_label,block_true_label];
end
block_true_label = tmp_label;clear tmp_label;












