function [subject] = get_subject_from_filename(filename_list)
%�������ܣ��������ļ���Ϣ�������ض��ṹ������
%filename_list �������ļ�����ɵ�cell�б�nii�ļ�����������������ļ���readme
%����һ����Ϊ��λ����Ϣ�ṹ��
%id��ʾ������ţ�v_id��ʾ������ct���
%v_m_id��ʾ��Ӧ��mask��ţ���һ������cell����v_id����Ŷ�Ӧ

%�����ṹ�忪ʼ�����ݽṹ��==================================================
%���������������ţ�������ŴӼ��������ɣ������ǲ���������ţ�
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
    subject(i).id = subject_id_list(i);%����ID��
    subject(i).v_id = [];%��ǰ�������˾�����ID��
    subject(i).a_id = [];%��ǰ�������˶�����ID��
    subject(i).n_id = [];%��ǰ��������ƽɨ��ID��
    subject(i).v_m_id = {};%��ǰ�������˾�����ID��
    subject(i).a_m_id = {};%��ǰ�������˶�����ID��
    subject(i).n_m_id = {};%��ǰ��������ƽɨ��ID��
    subject(i).v_tumor = {};%��������CT����
    subject(i).v_tumor_mask = {};%��������mask����
    subject(i).v_tumor_size = [];%��Ӧ�����Ĵ��Դ�С
    subject(i).v_tumor_size_all = [];%����������С֮��
    
end
%�����ṹ����������ݽṹ��==================================================


%����filename_list�����ڶ�id����ṹ�����顣
for i = 1:length(filename_list)
    tmp_filename = filename_list{i};
    index_ = find(tmp_filename == '_');
    %����ֻ��mask�ļ����ɣ�����Ҫ��ct����Ϊ��mask�϶��ж�Ӧct
    if length(index_) > 1
        subject_id = str2double(tmp_filename(2:index_(1)-1));
        index_id =  find(subject_id_list == subject_id);
        sec_name = tmp_filename(index_(1)+1);
        sec_id = str2double(tmp_filename(index_(1)+2:index_(2)-1)); %��ö���id
        if sec_name == 'a'
            if ~ismember(sec_id,subject(index_id).a_id)
                subject(index_id).a_id(end+1)=sec_id;%vid��ӵ���Ӧ�����ṹ����
            end
        elseif sec_name == 'v'
            if ~ismember(sec_id,subject(index_id).v_id)
                subject(index_id).v_id(end+1)=sec_id;%vid��ӵ���Ӧ�����ṹ����
            end
        elseif sec_name == 'n'
            if ~ismember(sec_id,subject(index_id).n_id)
                subject(index_id).n_id(end+1)=sec_id;%vid��ӵ���Ӧ�����ṹ����
            end
            %����������������ô��
        end
    end
end

%for v ����
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

%for a ����
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

%for n ����
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

