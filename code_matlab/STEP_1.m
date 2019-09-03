%% ����nii��Ϣ�����������ݱ������ض��ṹ�������У����ֱ𱣴没�����ݵ�����mat�ļ���
%% ��ʼ��===============================================================
clc;
clear;
%% ����·������H:\@data_NENs_recurrence\PNENs\data\a\0nii
nii_path =           'H:\@data_NENs_recurrence\PNENs\data\0nii\a';
adjust_window =            [-30,280];
nii_othermode_path = 'H:\@data_NENs_recurrence\PNENs\data\0nii\v';
adjust_window4_othermode = [-20,200];

save_path =          'H:\@data_NENs_recurrence\PNENs\data\1mat';

extend_pixel = 10;%�ٳ�����ʱ��������������
contain_orthermode = true;
disp(strcat('is cantain othermode ? :',num2str(contain_orthermode)));





%% ��ȡ������Ϣ����ʼ���ṹ������
nii_filename_list = dir(strcat(nii_path,filesep,'*.nii'));
filename_list = cell(length(nii_filename_list),1);
for i = 1:length(nii_filename_list)
    tmp_filename = nii_filename_list(i).name;
    filename_list{i}=tmp_filename(1:end-4);
end
subject = get_subject_from_filename(filename_list,contain_orthermode);
%% ���������nii�ļ�
%���ݲ�����Ϣ�������в�����mask�ٳ�����ÿһ��������������һ���ļ���
%��Ϣ������subject�ṹ�������Ӧ��Ϣ��������������Ӧ��������������������������������֮�͡�
%��ע��鿴���ݴ�����
for i = 1:length(subject)
    subject_ins =  subject(i);%�������˵Ľṹ�壬����������ձ���ĵ�������mat�ļ�
    disp(['@ doing with subject:',num2str(subject(i).id),' ...']);
    
    %�����������ģ̬������ǰ��������������λ��modeflag  {}{}{}{}
    if contain_orthermode
        subject_ins.tumor_othermode = {};
        subject_ins.tumor_mask_othermode = {};
    end
    
    
    
    %ͳ�Ƹò���������������
    num_tumor = 0;
    for ii = 1:length(subject_ins.v_m_id)
        num_tumor=num_tumor+length(subject_ins.v_m_id{ii});
    end
    disp(['tumors num :',num2str(num_tumor)]);
    subject(i).num_tumor = num_tumor;%�������������浽�ṹ��������
    subject_ins.num_tumor = num_tumor;
    
    
    tumor_index = 0;
    %��ȡCT
    for ii = 1:length(subject(i).v_id)
        sec_id = subject(i).v_id(ii);
        CT_file_name =strcat('s',num2str(subject(i).id),'_','v',num2str(sec_id),'.nii');
        Vref = spm_vol(strcat(nii_path,filesep,CT_file_name));
        Data_CT = spm_read_vols(Vref);%�������CTͼ�񣬽���������CT���ٳ�
        tmp_mask_id_list = subject_ins.v_m_id{ii};%��ȡ��Ӧ������mask id ����
        
        %�����������mode���������mode��CTҲ��ȡ����
        if contain_orthermode
            Vref = spm_vol(strcat(nii_othermode_path,filesep,CT_file_name));
            Data_CT_othermode = spm_read_vols(Vref);%�������CTͼ�񣬽���������CT���ٳ�
        end
        
        %�ڶ�Ӧ������mask id ������ѭ�����ٳ���Ԥ���������浽�ṹ��
        for iii = 1:length(tmp_mask_id_list)
            tumor_index = tumor_index+1;
            disp(['loading tumor',num2str(tumor_index),'...']);
            trd_id = tmp_mask_id_list(iii);
            
            % ��ȡ����mask
            mask_file_name =strcat('s',num2str(subject(i).id),'_','v',num2str(sec_id),'_m',num2str(trd_id),'.nii');
            Vref = spm_vol(strcat(nii_path,filesep,mask_file_name));
            Data_mask = spm_read_vols(Vref);
            
            %�����������mode���������mode��maskҲ��ȡ����
            if contain_orthermode
                Vref = spm_vol(strcat(nii_othermode_path,filesep,mask_file_name));
                Data_mask_othermode = spm_read_vols(Vref);%�������CTͼ�񣬽���������CT���ٳ�
            end
            
            % ���ݴ��������ڿ�ʼ ============================
            % ==============================================
            % ==============================================
            % ==============================================
            % ==============================================
            % flow 1:ʹ�ÿٳ�����������mask��������CT�пٳ�
            [tumor_mat, mask_mat, tumor_size,or_shape] = get_region_from_mask(Data_CT, Data_mask, extend_pixel);%������ǰ��CT������mask
            if contain_orthermode
                [tumor_mat_other, mask_mat_other,~,~] = get_region_from_mask(Data_CT_othermode, Data_mask_othermode, extend_pixel);
                %��Ϊ����������roi��һ����������Ҫ������resizeΪһ����С
                tumor_mat_other = imresize3(tumor_mat_other,or_shape,'cubic');
                mask_mat_other = imresize3(mask_mat_other,or_shape,'cubic');
            end
            
            % flow 2:Ԥ������������λ
            tumor_mat = tumor_preprocess(tumor_mat,'window_change',adjust_window);
            if contain_orthermode
                % �����ڿ�����Ҫ��ͬ�Ĵ���λ��:9 220 [200 -20]
                tumor_mat_other = tumor_preprocess(tumor_mat_other,'window_change',adjust_window4_othermode);
            end

            
            % flow 3:Ԥ�������Թ�һ��
            tumor_mat = tumor_preprocess(tumor_mat,'Linear_normalization');
            if contain_orthermode
                tumor_mat_other = tumor_preprocess(tumor_mat_other,'Linear_normalization');
            end
            % ���ݴ��������ڽ��� ============================
            % ==============================================
            % ==============================================
            % ==============================================
            % ==============================================

            
            
            %���浽�ṹ��
            subject_ins.tumor{end+1}=tumor_mat;
            subject_ins.tumor_mask{end+1}=mask_mat;
            subject_ins.tumor_shape{end+1}=or_shape;
            subject_ins.tumor_size(end+1)=tumor_size;%���������������������
            
            if contain_orthermode
                subject_ins.tumor_othermode{end+1}=tumor_mat_other;
                subject_ins.tumor_mask_othermode{end+1}=mask_mat_other;
            end
            

        end
        
    end
    % ��һ������ȫ������������֮�󣬴���ò���������������
    subject_ins.tumor_size_all = sum(subject_ins.tumor_size);
    % ����ṹ��subject_temp��ָ���ļ���
    disp('saving...');
    save(strcat(save_path,filesep,num2str(subject_ins.id)),'subject_ins');   
    
    % ˳�������Ҳ���浽�����õ��Ǹ��ṹ��������
    subject(i).tumor_size = subject_ins.tumor_size;
    subject(i).tumor_size_all = subject_ins.tumor_size_all;
    subject(i).tumor_shape = subject_ins.tumor_shape;

end

mkdir(strcat(save_path,filesep,'subject_all'));
save(strcat(save_path,filesep,'subject_all',filesep,'subject'),'subject'); 















