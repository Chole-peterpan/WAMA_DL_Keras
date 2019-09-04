function [subject] = get_subject_from_filename(filename_list,contain_orthermode)
%�������ܣ��������ļ���Ϣ�������ض��ṹ������
%filename_list �������ļ�����ɵ�cell�б�nii�ļ�����������������ļ���readme
%����һ����Ϊ��λ����Ϣ�ṹ��
%id��ʾ������ţ�v_id��ʾ������ct���
%v_m_id��ʾ��Ӧ��mask��ţ���һ������cell����v_id����Ŷ�Ӧ


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


%��ʼ���ṹ�忪ʼ
subject = [];
for i =1:length(subject_id_list)
    subject(i).id = subject_id_list(i);%����ID��
    
    % ÿ�������в�ͬ����ÿ���������ɨ����
    % ����һ��ɨ�����������һ����ɨ���ϸ����ڶ���ɨ���¸��������������ͬһ����������ж���ļ�
    subject(i).v_id = [];%��ǰ��������ĳ����ͼ��ID��
    
    % ÿ�������ÿ���ļ������ж�Ӧ��mask��Ҳ������������ÿ��CTid�ţ������ܶ�Ӧ�Զ������
    subject(i).v_m_id = {};% ��ǰ����ĳ�����Ӧ����maskID��
    % ���ھ����ڣ���ʵ��������һ���ģ�����ֻ����һ�����ӣ�
    subject(i).voxel_size = {};%�������س���ߵ�cell
%     subject(i).tumor = {};%��������CT����
%     subject(i).tumor_mask = {};%��������mask����
    subject(i).tumor_shape = {};%��������de��������״
    subject(i).tumor_size = [];%��Ӧ�����Ĵ��Դ�С
    subject(i).tumor_size_all = [];%����������С֮��
    
    % �Ƿ��������mode��flag��������������������λ
    subject(i).othermode = contain_orthermode;
    
end


%����filename_list�����ڶ�id����ṹ�����顣
for i = 1:length(filename_list)
    tmp_filename = filename_list{i};
    index_ = find(tmp_filename == '_');
    %����ֻ��mask�ļ����ɣ�����Ҫ��ct����Ϊ��mask�϶��ж�Ӧct
    if length(index_) > 1
        subject_id = str2double(tmp_filename(2:index_(1)-1));
        index_id =  find(subject_id_list == subject_id);
        sec_id = str2double(tmp_filename(index_(1)+2:index_(2)-1)); %��ö���id 
        if ~ismember(sec_id,subject(index_id).v_id)
            subject(index_id).v_id(end+1)=sec_id;%��ӵ���Ӧ�����ṹ����
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

