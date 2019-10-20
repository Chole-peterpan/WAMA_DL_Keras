
%% 加载subject文件，并制作id到label的映射,以及每个label中各个类的名称
workspaces = load(strcat(block_mat_path,filesep,'subject',filesep,'subject.mat'));
subject = workspaces.subject;%记录所有统计信息的mat文件，为结构体数组。
all_label = workspaces.xls_data(:,2:end);
% 计算每一种label共包含多少类，比如类别为分级则包含3类，类别为男女两类
per_label_class_num = [];
per_label_index = {};
for i = 1:size(all_label,2)
    tmp_all_class = unique(all_label(:,i));
    per_label_index{end+1} = tmp_all_class;
    per_label_class_num(end+1) = length(tmp_all_class);
end
%% 转换文件：每个h5文件中存放的分别是，data，label，labelnum_perclass,labelindex_perclass
filename_list = dir(strcat(block_mat_path,filesep,'*.mat'));

parfor ii = 1:length(filename_list)
    % 读取文件名
    filename = filename_list(ii,1).name;
   
    % 读取block
    data_path=strcat(block_mat_path,filesep,filename);
    workspaces = load(data_path);
    data = workspaces.block;
    
    % 读取label
    label = workspaces.label;    

    % 处理数据（限定输出format），使之shape相同
    [aug_data,~] = aug43D(data,  augdict);

    % 构建储存路径
    finalpath = strcat(h5_savepath,filesep,filename(1:end-4),'.h5');
    
    % 储存label
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
