%% 储存随机数种子，以便复现实验
augdict.s = rng;

%% 加载病人信息
workspaces = load(strcat(block_mat_path,filesep,'subject',filesep,'subject.mat'));
subject = workspaces.subject;%记录所有统计信息的mat文件，为结构体数组。

%% 计算每一类样本扩增的总数量
per_class_aug_num = floor(augdict.aug_num*augdict.a_b_ratio/sum(augdict.a_b_ratio))+1;

%% 计算每类中每个样本应扩增的数量
all_label = workspaces.xls_data(:,2:end);
% 计算每一种label共包含多少类，比如类别为分级则包含3类，类别为男女两类
per_label_class_num = [];
per_label_index = {};
for i = 1:size(all_label,2)
    tmp_all_class = unique(all_label(:,i));
    per_label_index{end+1} = tmp_all_class;
    per_label_class_num(end+1) = length(tmp_all_class);
end

%提取指定label的label值
iden_label = workspaces.xls_data(:,(augdict.balance_label_index)+1);
%计算各类样本的数量
per_class_person = [];
for i = 1:length(per_label_index{augdict.balance_label_index})
    per_class_person(end+1) = length(find(iden_label == per_label_index{augdict.balance_label_index}(i)));
end

%计算每一类中每个样本应扩增的数量
per_class_per_person_aug_num = per_class_aug_num./per_class_person;
pcppa_num = per_class_per_person_aug_num;%缩写

%% 接下来以病人为单位进行操作，计算每个病人的每个block应该被扩增多少块,并保存到结构体数组中
for i = 1:length(subject)
    subject_ins = subject(i);
    blocks = subject_ins.blocks_num_all;
    
    % 获取该样本应该扩增的数量
    tmp_label = subject_ins.or_label(augdict.balance_label_index);
    person_aug_num = pcppa_num(per_label_index{augdict.balance_label_index}==tmp_label);
    
    % 计算该样本每个块应该扩增的数量
    per_block_aug_num = ceil(person_aug_num/blocks);
    
    %将信息储存到病人结构体数组中
    subject(i).per_block_aug_num = per_block_aug_num;
    subject(i).per_tumor_aug_num = {};
    all_aug_num = 0;
    for ii = 1:length(subject(i).blocks_num_per_tumor)
        subject(i).per_tumor_aug_num{ii}= per_block_aug_num*(subject(i).blocks_num_per_tumor{ii});
        all_aug_num  = all_aug_num + sum(subject(i).per_tumor_aug_num{ii});
    end
    
    subject(i).all_aug_num = all_aug_num;
end


%% 之后循环block列表，每读取一个block，就找到他的id对应的per_block_augnum，之后扩增就完事了，记得把扩增细节保存到block_detail
block_detail = [];

filename_list = dir(strcat(block_mat_path,filesep,'*.mat'));
parfor ii = 1:length(filename_list)
    % 读取某一个block的文件名
    filename = filename_list(ii,1).name;
    data_path=strcat(block_mat_path,filesep,filename);
    
    tmpcell = strsplit(filename,'_');
    id = str2double(tmpcell{1}(2:end));
    subject_ins = subject([subject.id] == id);

    

    workspaces = load(data_path);
    data = workspaces.block;
    per_block_aug_num =  subject_ins.per_block_aug_num;
    
    
    for iii = 1:per_block_aug_num
        % 扩增
        [aug_data,aug_detail] = aug43D(data,  augdict);
              
        % 保存 到 detail里面
        tmp_name = strcat(filename(1:end-4),'_e',num2str(iii));
        
        % 如果用parfor的话，下面这两句就不能用
%         block_detail(end+1).name = tmp_name;
%         block_detail(end).aug_detail = aug_detail;
        
        % 构建最终保存文件名
        write_name = strcat(tmp_name,'.h5');
        fprintf('aug_num:%d    aug_file_name: %s\n',iii,write_name);
        finalpath = strcat(augdict.mat_savepath,filesep,write_name);%最终储存路径
        disp(finalpath);
        
        
        % 储存label
        label = workspaces.label;
        h5create(finalpath, '/label', size(label),'Datatype','single');
        h5write(finalpath, '/label', label);
        
        % 储存data,写个for循环把cell都存进去，统一命名，模态1的索引就是mode1，模态2就是mode2，以此类推
        for i = 1:length(aug_data)
            tmp_index = ['/mode',num2str(i)];
            h5create(finalpath, tmp_index, size(aug_data{i}),'Datatype','single');
            h5write(finalpath, tmp_index, aug_data{i});
        end
        
        % 储存labelnum_perclass,labelindex_perclass
        h5create(finalpath, '/per_label_class_num', size(per_label_class_num),'Datatype','single');
        h5write(finalpath, '/per_label_class_num', per_label_class_num);
        for i = 1:length(per_label_index)
            tmp_index = ['/per_label_index',num2str(i)];
            h5create(finalpath, tmp_index, size(per_label_index{i}),'Datatype','single');
            h5write(finalpath, tmp_index, per_label_index{i});
        end
    end
end

mkdir(strcat(augdict.mat_savepath,filesep,'subject'));
save(strcat(augdict.mat_savepath,filesep,'subject',filesep,'subject.mat'),'subject','augdict','block_detail');







