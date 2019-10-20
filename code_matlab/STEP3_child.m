
%% ����subject�ļ���������id��label��ӳ��,�Լ�ÿ��label�и����������
workspaces = load(strcat(block_mat_path,filesep,'subject',filesep,'subject.mat'));
subject = workspaces.subject;%��¼����ͳ����Ϣ��mat�ļ���Ϊ�ṹ�����顣
all_label = workspaces.xls_data(:,2:end);
% ����ÿһ��label�����������࣬�������Ϊ�ּ������3�࣬���Ϊ��Ů����
per_label_class_num = [];
per_label_index = {};
for i = 1:size(all_label,2)
    tmp_all_class = unique(all_label(:,i));
    per_label_index{end+1} = tmp_all_class;
    per_label_class_num(end+1) = length(tmp_all_class);
end
%% ת���ļ���ÿ��h5�ļ��д�ŵķֱ��ǣ�data��label��labelnum_perclass,labelindex_perclass
filename_list = dir(strcat(block_mat_path,filesep,'*.mat'));

parfor ii = 1:length(filename_list)
    % ��ȡ�ļ���
    filename = filename_list(ii,1).name;
   
    % ��ȡblock
    data_path=strcat(block_mat_path,filesep,filename);
    workspaces = load(data_path);
    data = workspaces.block;
    
    % ��ȡlabel
    label = workspaces.label;    

    % �������ݣ��޶����format����ʹ֮shape��ͬ
    [aug_data,~] = aug43D(data,  augdict);

    % ��������·��
    finalpath = strcat(h5_savepath,filesep,filename(1:end-4),'.h5');
    
    % ����label
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
