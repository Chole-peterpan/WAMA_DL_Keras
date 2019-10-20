%% �����ļ�������������Ϣ
nii_filename_list = dir(strcat(data_path{1},filesep,'*.nii'));
filename_list = cell(length(nii_filename_list),1);
for i = 1:length(nii_filename_list)
    tmp_filename = nii_filename_list(i).name;
    filename_list{i}=tmp_filename(1:end-4);
end
subject = get_subject_from_filename(filename_list, data_path);

%% ���財��label
xls_data = xlsread(label_file, 1);
for i = 1:length(subject)
    subject(i).or_label = xls_data((xls_data(:,1)==subject(i).id),2:end);
end

%% ���������nii�ļ�
%���ݲ�����Ϣ�������в�����mask��roi�ٳ�����ÿһ��������������һ���ļ���
%��Ϣ������subject�ṹ�������Ӧ��Ϣ��������������Ӧ��������������������������������֮�͡�
%��ע��鿴���ݴ�����!!!!!!!
for i = 1:length(subject) % i �ǲ��˵�����
    subject_ins =  subject(i);%�������˵Ľṹ�壬����������ձ���ĵ�������mat�ļ�
    
    % ��ʾ����
    if rem(i,2)==0
        disp(['/(@.@)/ doing with subject:',num2str(subject_ins.id),' ...']);
    else
        disp(['\(@.@)\ doing with subject:',num2str(subject_ins.id),' ...']);
    end
    
    % �����ļ�����Ҫ���������������ļ�subject_ins����Ҫ��������Ҫ�½�
    subject_ins.tumor = {};%��������CT����
    subject_ins.tumor_mask = {};%��������mask����
    
    %ͳ�Ƹò���������������
    num_tumor = 0;
    for ii = 1:length(subject_ins.v_m_id)
        num_tumor=num_tumor+length(subject_ins.v_m_id{ii});
    end
    disp(['tumors num :',num2str(num_tumor)]);
    subject_ins.num_tumor = num_tumor;
    
    % ����ÿһ��ģ̬
    for ii = 1:subject_ins.othermode  % ii ��ģ̬������
        disp(['processing mode ',num2str(ii)]);
        % ÿһ��ģ̬������������������
        tumor_index = 0;
        % ����һ��Ŀ¼cell:ģ̬����
        subject_ins.tumor{end+1} = {};
        subject_ins.tumor_mask{end+1} = {};
        subject_ins.voxel_size{end+1} = {};
        subject_ins.tumor_shape{end+1} = {};
        subject_ins.tumor_location{end+1} = {};
        subject_ins.tumor_size{end+1} = {};
        
        % ����ÿһ��/��CT��MRIɨ���
        for iii = 1:length(subject_ins.v_id) % iii ��ɨ��ε�����
            % ��������Ŀ¼cell��CT�ֶμ���
            subject_ins.tumor{end}{end+1} = {};
            subject_ins.tumor_mask{end}{end+1} = {};
            % subject_ins.voxel_size{end}{end+1} = {};% �����CT����͹��ˣ����Բ���Ҫ����cellĿ¼��
            subject_ins.tumor_shape{end}{end+1} = {};
            subject_ins.tumor_location{end}{end+1} = {};
            subject_ins.tumor_size{end}{end+1} = {};
                        
            sec_id = subject_ins.v_id(iii);% ��ȡ�ֶ�CT��id
            % ��ȡCT
            CT_file_name =strcat('s',num2str(subject_ins.id),'_v',num2str(sec_id),'.nii');
            Vref = spm_vol(strcat(data_path{ii},filesep,CT_file_name));
            Data_CT = spm_read_vols(Vref);%�������CTͼ�񣬽���������CT���ٳ�
            %����ÿ��CT������size
            voxel_size = abs([Vref.mat(1,1),Vref.mat(2,2),Vref.mat(3,3)]);
            subject_ins.voxel_size{end}{end+1} = voxel_size;
            
            
            % ����ÿһ������
            tmp_mask_id_list = subject_ins.v_m_id{iii};%��ȡ��Ӧ������mask id�б�
            for iiii = 1:length(tmp_mask_id_list) % iiii ������������
                tumor_index = tumor_index+1;
                disp(['loading mode :',num2str(ii),', tumor:',num2str(tumor_index),'...']);
                trd_id = tmp_mask_id_list(iiii);
                
                % ��ȡ����mask
                mask_file_name =strcat('s',num2str(subject_ins.id),'_v',num2str(sec_id),'_m',num2str(trd_id),'.nii');
                Vref = spm_vol(strcat(data_path{ii},filesep,mask_file_name));
                Data_mask = spm_read_vols(Vref);
                
                

                
                % ���ݴ��������ڿ�ʼ ============================
                % ==============================================
                % ==============================================
                % ==============================================
                % ==============================================
                % flow 1:ʹ�ÿٳ�����������mask��������CT�пٳ�
                extend_pixel = floor(extend_length / voxel_size(1));
                [tumor_mat, mask_mat, tumor_size,or_shape,or_location] = get_region_from_mask(Data_CT, Data_mask, extend_pixel);%������ǰ��CT������mask
              
                
                % 3Dresize��Ĭ��resize����һ��ģ̬��������С
                if ii ~= 1
                    tumor_mat = imresize3(tumor_mat,subject_ins.tumor_shape{1}{iii}{iiii},'cubic');
                    mask_mat = imresize3(mask_mat,or_shape,'cubic');
                end
                                
                % flow 2:Ԥ������������λ
                tumor_mat = CT_data_preprocess(tumor_mat,'window_change',data_window{ii});
                
                % flow 3:Ԥ�����ռ������ز���
                if resample_flag
                    tumor_mat = CT_data_preprocess(tumor_mat, 'voxel_dim_resampling', voxel_size, adjust_voxelsize);
                    mask_mat = CT_data_preprocess(mask_mat, 'voxel_dim_resampling', voxel_size, adjust_voxelsize);
                end

                % flow 4:Ԥ�������Թ�һ��
                tumor_mat = CT_data_preprocess(tumor_mat,'Linear_normalization');
                % ���ݴ��������ڽ��� ============================
                % ==============================================
                % ==============================================
                % ==============================================
                % ==============================================
                
                %�������ָ�굽�ṹ��
                subject_ins.tumor{end}{end}{end+1}=tumor_mat;
                subject_ins.tumor_mask{end}{end}{end+1}=mask_mat;
                subject_ins.tumor_shape{end}{end}{end+1}=or_shape;
                subject_ins.tumor_location{end}{end}{end+1}=or_location;
                subject_ins.tumor_size{end}{end}{end+1}=tumor_size;%��������������������� 
            end   
        end

    end
    % ���������������Ĭ�ϼ����һ��ģ̬�����
    tumor_size_all = 0;
    for n = 1:length(subject_ins.tumor_size{1})
        for nn = 1:length(subject_ins.tumor_size{1}{n})
            tumor_size_all = tumor_size_all + (subject_ins.tumor_size{1}{n}{nn});
        end
    end
    subject_ins.tumor_size_all = tumor_size_all;
    % ˳�������Ҳ���浽�����õ��Ǹ��ṹ��������
    subject(i).num_tumor = subject_ins.num_tumor;
    subject(i).tumor_size = subject_ins.tumor_size;
    subject(i).tumor_size_all = subject_ins.tumor_size_all;
    subject(i).tumor_shape = subject_ins.tumor_shape;
    subject(i).voxel_size = subject_ins.voxel_size;
    subject(i).tumor_location = subject_ins.tumor_location;
    

    % ����ṹ��subject_temp��ָ���ļ���
    disp('saving...');
    save(strcat(save_path,filesep,num2str(subject_ins.id)),'subject_ins');   

end

mkdir(strcat(save_path,filesep,'subject_all'));
save(strcat(save_path,filesep,'subject_all',filesep,'subject'),'subject',...
    'data_path','data_window','xls_data','resample_flag','extend_length','adjust_voxelsize'); 


