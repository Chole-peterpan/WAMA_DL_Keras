%% ������������ӣ��Ա㸴��ʵ��
augdict.s = rng;

%% ���ز�����Ϣ
workspaces = load(strcat(block_mat_path,filesep,'subject',filesep,'subject.mat'));
subject = workspaces.subject;%��¼����ͳ����Ϣ��mat�ļ���Ϊ�ṹ�����顣

%% ����ÿһ������������������
per_class_aug_num = floor(augdict.aug_num*augdict.a_b_ratio/sum(augdict.a_b_ratio))+1;

%% ����ÿ����ÿ������Ӧ����������
all_label = workspaces.xls_data(:,2:end);
% ����ÿһ��label�����������࣬�������Ϊ�ּ������3�࣬���Ϊ��Ů����
per_label_class_num = [];
per_label_index = {};
for i = 1:size(all_label,2)
    tmp_all_class = unique(all_label(:,i));
    per_label_index{end+1} = tmp_all_class;
    per_label_class_num(end+1) = length(tmp_all_class);
end

%��ȡָ��label��labelֵ
iden_label = workspaces.xls_data(:,(augdict.balance_label_index)+1);
%�����������������
per_class_person = [];
for i = 1:length(per_label_index{augdict.balance_label_index})
    per_class_person(end+1) = length(find(iden_label == per_label_index{augdict.balance_label_index}(i)));
end

%����ÿһ����ÿ������Ӧ����������
per_class_per_person_aug_num = per_class_aug_num./per_class_person;
pcppa_num = per_class_per_person_aug_num;%��д

%% �������Բ���Ϊ��λ���в���������ÿ�����˵�ÿ��blockӦ�ñ��������ٿ�,�����浽�ṹ��������
for i = 1:length(subject)
    subject_ins = subject(i);
    blocks = subject_ins.blocks_num_all;
    
    % ��ȡ������Ӧ������������
    tmp_label = subject_ins.or_label(augdict.balance_label_index);
    person_aug_num = pcppa_num(per_label_index{augdict.balance_label_index}==tmp_label);
    
    % ���������ÿ����Ӧ������������
    per_block_aug_num = ceil(person_aug_num/blocks);
    
    %����Ϣ���浽���˽ṹ��������
    subject(i).per_block_aug_num = per_block_aug_num;
    subject(i).per_tumor_aug_num = {};
    all_aug_num = 0;
    for ii = 1:length(subject(i).blocks_num_per_tumor)
        subject(i).per_tumor_aug_num{ii}= per_block_aug_num*(subject(i).blocks_num_per_tumor{ii});
        all_aug_num  = all_aug_num + sum(subject(i).per_tumor_aug_num{ii});
    end
    
    subject(i).all_aug_num = all_aug_num;
end


%% ֮��ѭ��block�б�ÿ��ȡһ��block�����ҵ�����id��Ӧ��per_block_augnum��֮�������������ˣ��ǵð�����ϸ�ڱ��浽block_detail
block_detail = [];

filename_list = dir(strcat(block_mat_path,filesep,'*.mat'));
parfor ii = 1:length(filename_list)
    % ��ȡĳһ��block���ļ���
    filename = filename_list(ii,1).name;
    data_path=strcat(block_mat_path,filesep,filename);
    
    tmpcell = strsplit(filename,'_');
    id = str2double(tmpcell{1}(2:end));
    subject_ins = subject([subject.id] == id);

    

    workspaces = load(data_path);
    data = workspaces.block;
    per_block_aug_num =  subject_ins.per_block_aug_num;
    
    
    for iii = 1:per_block_aug_num
        % ����
        [aug_data,aug_detail] = aug43D(data,  augdict);
              
        % ���� �� detail����
        tmp_name = strcat(filename(1:end-4),'_e',num2str(iii));
        
        % �����parfor�Ļ�������������Ͳ�����
%         block_detail(end+1).name = tmp_name;
%         block_detail(end).aug_detail = aug_detail;
        
        % �������ձ����ļ���
        write_name = strcat(tmp_name,'.h5');
        fprintf('aug_num:%d    aug_file_name: %s\n',iii,write_name);
        finalpath = strcat(augdict.mat_savepath,filesep,write_name);%���մ���·��
        disp(finalpath);
        
        
        % ����label
        label = workspaces.label;
        h5create(finalpath, '/label', size(label),'Datatype','single');
        h5write(finalpath, '/label', label);
        
        % ����data,д��forѭ����cell�����ȥ��ͳһ������ģ̬1����������mode1��ģ̬2����mode2���Դ�����
        for i = 1:length(aug_data)
            tmp_index = ['/mode',num2str(i)];
            h5create(finalpath, tmp_index, size(aug_data{i}),'Datatype','single');
            h5write(finalpath, tmp_index, aug_data{i});
        end
        
        % ����labelnum_perclass,labelindex_perclass
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







