function [subject] = get_subject_from_filename(filename_list,contain_orthermode)
%函数功能：整理病例文件信息并返回特定结构体数组
%filename_list 是所有文件名组成的cell列表，nii文件命名规则见本函数文件夹readme
%返回一并认为单位的信息结构体
%id表示病人序号，v_id表示动脉期ct序号
%v_m_id表示对应的mask序号，是一个矩阵cell，与v_id中序号对应


%获得所有样本的序号（样本序号从几到几都可，可以是不连续的序号）
subject_id_list = [];
for i = 1:length(filename_list)
    tmp_filename = filename_list{i};
    index_ = find(tmp_filename == '_');
    subject_id = str2double(tmp_filename(2:index_(1)-1));
    if ~ismember(subject_id,subject_id_list)
        subject_id_list(end+1) = subject_id;
    end
end
subject_id_list = subject_id_list';
subject_id_list = sort(subject_id_list);


%初始化结构体开始
subject = [];
for i =1:length(subject_id_list)
    subject(i).id = subject_id_list(i);%病例ID号
    
    % 每个病人有不同期像，每个期像可能扫描多次
    % 比如一次扫不完上身，则第一次先扫描上腹，第二次扫描下腹，这样就造成了同一个期像可能有多个文件
    subject(i).v_id = [];%提前构建病人某期像图像ID号
    
    % 每个期像的每个文件，都有对应的mask（也就是肿瘤），每个CTid号，都可能对应对多个肿瘤
    subject(i).v_m_id = {};% 提前构建某期像对应肿瘤maskID号
    % 对于静脉期（其实三个期是一样的，所以只构建一个足矣）
    subject(i).voxel_size = {};%储存体素长宽高的cell
%     subject(i).tumor = {};%储存肿瘤CT矩阵
%     subject(i).tumor_mask = {};%储存肿瘤mask矩阵
    subject(i).tumor_shape = {};%储存肿瘤de粗略略形状
    subject(i).tumor_size = [];%对应肿瘤的粗略大小
    subject(i).tumor_size_all = [];%所有肿瘤大小之和
    
    % 是否包含其他mode的flag，如果包含，则后续会置位
    subject(i).othermode = contain_orthermode;
    
end


%遍历filename_list，将第二id导入结构体数组。
for i = 1:length(filename_list)
    tmp_filename = filename_list{i};
    index_ = find(tmp_filename == '_');
    %这里只看mask文件即可，不需要看ct，因为有mask肯定有对应ct
    if length(index_) > 1
        subject_id = str2double(tmp_filename(2:index_(1)-1));
        index_id =  find(subject_id_list == subject_id);
        sec_id = str2double(tmp_filename(index_(1)+2:index_(2)-1)); %获得二级id 
        if ~ismember(sec_id,subject(index_id).v_id)
            subject(index_id).v_id(end+1)=sec_id;%添加到对应病例结构体中
        end
    end
end

for ii = 1:length(subject)
    sub_id = num2str(subject(ii).id);
    for iii = 1:length(subject(ii).v_id)
        sec_id = num2str(subject(ii).v_id(iii));
        tmp_str = strcat('s',sub_id,'_','v',sec_id,'_');
        tmp_v_m_id = [];
        for iiii = 1:length(filename_list)
            tmp_filename = filename_list{iiii};
            index_ = find(tmp_filename == '_');
            if strcmp(tmp_str,tmp_filename(1:index_(end))) == 1
                tmp_v_m_id(end+1) = str2double(tmp_filename(index_(end)+2:end));
            end
        end
        subject(ii).v_m_id{iii}=tmp_v_m_id;
    end
end


end

