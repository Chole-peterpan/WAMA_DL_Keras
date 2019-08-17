%����nii��Ϣ�����������ݱ������ض��ṹ���У����ֱ𱣴没�����ݵ�mat�ļ���
%ͬʱ�������в�����Ϣ�Թ��պ�ͳ��
%��ʼ��===============================================================
clc;
clear;
%���ò���
save_path = 'G:\@data_NENs_recurrence\PNENs\data\a\1mat';
nii_path = 'G:\@data_NENs_recurrence\PNENs\data\a\0nii';
%=====================================================================
%��ȡ������Ϣ
nii_filename_list = dir(strcat(nii_path,filesep,'*.nii'));
filename_list = cell(length(nii_filename_list),1);
for i = 1:length(nii_filename_list)
    tmp_filename = nii_filename_list(i).name;
    filename_list{i}=tmp_filename(1:end-4);
end
subject = get_subject_from_filename(filename_list);

%��������Ŀǰֻ�о����ڣ���ֻ�Ծ����ڽ��в���,
%���ݲ�����Ϣ�������в�����mask�ٳ�����ÿһ��������������һ���ļ���
%��Ϣ������subject�ṹ�������Ӧ��Ϣ��������������Ӧ��������������������������������֮�͡�
%��һ������Ԥ�����ȿ۳�����Ԥ����Ҳ����֮������Ԥ�����ٿ۳�����
sec_name = 'v';
for i = 1:length(subject)
    subject_temp =  subject(i);%����������ձ����
    disp(['@ subject:',num2str(subject(i).id),'=====================']);
    
    %ͳ����������
    num_tumor = 0;
    for ii = 1:length(subject_temp.v_m_id)
        num_tumor=num_tumor+length(subject_temp.v_m_id{ii});
    end
    subject(i).v_num_tumor = num_tumor;%�������������浽�ṹ��������
    disp(['tumors num :',num2str(num_tumor)]);
    subject_temp.v_num_tumor = num_tumor;
    
    tumor_index = 0;
    %��ȡCT
    for ii = 1:length(subject(i).v_id)
        sec_id = subject(i).v_id(ii);
        CT_file_name =strcat('s',num2str(subject(i).id),'_',sec_name,num2str(sec_id),'.nii');
        Vref = spm_vol(strcat(nii_path,filesep,CT_file_name));
        Data_CT = spm_read_vols(Vref);%�������CTͼ�񣬽���������CT���ٳ�
        tmp_mat = subject_temp.v_m_id{ii};%��ȡ��Ӧ������mask id ����
        
        %�ڶ�Ӧ������mask id ������ѭ�����ٳ���Ԥ���������浽�ṹ��
        for iii = 1:length(tmp_mat)
            tumor_index = tumor_index+1;
            disp(['loading tumor',num2str(tumor_index)]);
            trd_id = tmp_mat(iii);
            mask_file_name =strcat('s',num2str(subject(i).id),'_',...
                sec_name,num2str(sec_id),'_m',num2str(trd_id),'.nii');
            
            Vref = spm_vol(strcat(nii_path,filesep,mask_file_name));
            Data_mask = spm_read_vols(Vref);%�������maskͼ�񣬽���������CT���ٳ�
            
            %ʹ�ÿٳ�����get_region_from_mask�ٳ���Ӧ����,���������
            [tumor_mat,mask_mat,tumor_size,~] = get_region_from_mask(Data_CT,Data_mask,5);%������ǰ��CT������mask
            %Ԥ������tumor_preprocess
            tumor_mat = tumor_preprocess(tumor_mat);
            %���浽�ṹ��
            subject_temp.v_tumor{end+1}=tumor_mat;
            subject_temp.v_tumor_mask{end+1}=mask_mat;
            subject_temp.v_tumor_size(end+1)=tumor_size;%���������������������
        end
        
    end
    subject_temp.v_tumor_size_all = sum(subject_temp.v_tumor_size);
    %����ṹ��subject_temp��ָ���ļ���
    disp('saving...');
    save(strcat(save_path,filesep,num2str(subject_temp.id)),'subject_temp');   
    
    subject(i).v_tumor_size = subject_temp.v_tumor_size;
    subject(i).v_tumor_size_all = subject_temp.v_tumor_size_all;
    
    
    
end

mkdir(strcat(save_path,filesep,'subject_all'));
save(strcat(save_path,filesep,'subject_all',filesep,'subject'),'subject'); 















