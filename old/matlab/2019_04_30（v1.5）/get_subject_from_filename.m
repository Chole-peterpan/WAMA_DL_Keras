function [subject] = get_subject_from_filename(filename_list)
%函数功能：整理病例文件信息并返回特定结构体数组
%filename_list 是所有文件名组成的cell列表，nii文件命名规则见本函数文件夹readme
%返回一并认为单位的信息结构体
%id表示病人序号，v_id表示动脉期ct序号
%v_m_id表示对应的mask序号，是一个矩阵cell，与v_id中序号对应

%构建结构体开始（数据结构）==================================================
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
subject = [];
for i =1:length(subject_id_list)
    subject(i).id = subject_id_list(i);%病例ID号
    subject(i).v_id = [];%提前构建病人静脉期ID号
    subject(i).a_id = [];%提前构建病人动脉期ID号
    subject(i).n_id = [];%提前构建病人平扫期ID号
    subject(i).v_m_id = {};%提前构建病人静脉期ID号
    subject(i).a_m_id = {};%提前构建病人动脉期ID号
    subject(i).n_m_id = {};%提前构建病人平扫期ID号
    subject(i).v_tumor = {};%储存肿瘤CT矩阵
    subject(i).v_tumor_mask = {};%储存肿瘤mask矩阵
    subject(i).v_tumor_size = [];%对应肿瘤的粗略大小
    subject(i).v_tumor_size_all = [];%所有肿瘤大小之和
    
end
%构建结构体结束（数据结构）==================================================


%遍历filename_list，将第二id导入结构体数组。
for i = 1:length(filename_list)
    tmp_filename = filename_list{i};
    index_ = find(tmp_filename == '_');
    %这里只看mask文件即可，不需要看ct，因为有mask肯定有对应ct
    if length(index_) > 1
        subject_id = str2double(tmp_filename(2:index_(1)-1));
        index_id =  find(subject_id_list == subject_id);
        sec_name = tmp_filename(index_(1)+1);
        sec_id = str2double(tmp_filename(index_(1)+2:index_(2)-1)); %获得二级id
        if sec_name == 'a'
            if ~ismember(sec_id,subject(index_id).a_id)
                subject(index_id).a_id(end+1)=sec_id;%vid添加到对应病例结构体中
            end
        elseif sec_name == 'v'
            if ~ismember(sec_id,subject(index_id).v_id)
                subject(index_id).v_id(end+1)=sec_id;%vid添加到对应病例结构体中
            end
        elseif sec_name == 'n'
            if ~ismember(sec_id,subject(index_id).n_id)
                subject(index_id).n_id(end+1)=sec_id;%vid添加到对应病例结构体中
            end
            %相加其他期像可以这么加
        end
    end
end

%for v 期像
sec_name = 'v';
for ii = 1:length(subject)
    sub_id = num2str(subject(ii).id);
    for iii = 1:length(subject(ii).v_id)
        sec_id = num2str(subject(ii).v_id(iii));
        tmp_str = strcat('s',sub_id,'_',sec_name,sec_id,'_');
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

%for a 期像
sec_name = 'a';%
for ii = 1:length(subject)
    sub_id = num2str(subject(ii).id);
    for iii = 1:length(subject(ii).a_id)%
        sec_id = num2str(subject(ii).a_id(iii));%
        tmp_str = strcat('s',sub_id,'_',sec_name,sec_id,'_');
        tmp_a_m_id = [];%
        for iiii = 1:length(filename_list)
            tmp_filename = filename_list{iiii};
            index_ = find(tmp_filename == '_');
            if strcmp(tmp_str,tmp_filename(1:index_(end))) == 1
                tmp_a_m_id(end+1) = str2double(tmp_filename(index_(end)+2:end));%
            end
        end
        subject(ii).a_m_id{iii}=tmp_a_m_id;%
    end
end

%for n 期像
sec_name = 'n';%
for ii = 1:length(subject)
    sub_id = num2str(subject(ii).id);
    for iii = 1:length(subject(ii).n_id)%
        sec_id = num2str(subject(ii).n_id(iii));%
        tmp_str = strcat('s',sub_id,'_',sec_name,sec_id,'_');
        tmp_n_m_id = [];%
        for iiii = 1:length(filename_list)
            tmp_filename = filename_list{iiii};
            index_ = find(tmp_filename == '_');
            if strcmp(tmp_str,tmp_filename(1:index_(end))) == 1
                tmp_n_m_id(end+1) = str2double(tmp_filename(index_(end)+2:end));%
            end
        end
        subject(ii).n_m_id{iii}=tmp_n_m_id;%
    end
end


end

