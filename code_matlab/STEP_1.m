%% 整理nii信息并将病人数据保存在特定结构体数组中，并分别保存病人数据到单个mat文件。
%% 初始化===============================================================
clc;
clear;
%% 设置路径参数H:\@data_NENs_recurrence\PNENs\data\a\0nii
nii_path =           'H:\@data_NENs_recurrence\PNENs\data_outside\0nii\a';
adjust_window =            [-25,285];
nii_othermode_path = 'H:\@data_NENs_recurrence\PNENs\data_outside\0nii\v';
adjust_window4_othermode = [-15,205];

save_path =          'H:\@data_NENs_recurrence\PNENs\data\flow2\1mat';

resample_flag = true;
extend_pixel = 20;%抠出肿瘤时外扩的体素数量
adjust_voxelsize = [0.5,0.5,0.5];%空间体素重采样的目标size
contain_orthermode = true;
disp(strcat('is cantain othermode ? :',num2str(contain_orthermode)));





%% 提取病例信息，初始化结构体数组
nii_filename_list = dir(strcat(nii_path,filesep,'*.nii'));
filename_list = cell(length(nii_filename_list),1);
for i = 1:length(nii_filename_list)
    tmp_filename = nii_filename_list(i).name;
    filename_list{i}=tmp_filename(1:end-4);
end
subject = get_subject_from_filename(filename_list,contain_orthermode);
%% 逐个处理病人nii文件
%根据病例信息，对所有病例的mask抠出，并每一个病例单独保存一个文件，
%信息包括：subject结构体数组对应信息，肿瘤个数，对应肿瘤的体积，单个病人所有肿瘤的体积之和。
%请注意查看数据处理流
for i = 1:length(subject)
    subject_ins =  subject(i);%单个病人的结构体，这个就是最终保存的单个病人mat文件
    disp(['@ doing with subject:',num2str(subject_ins.id),' ...']);
    
    
    % 汇总文件不需要储存肿瘤，单个文件才需要，所以需要新建
    subject_ins.tumor = {};%储存肿瘤CT矩阵
    subject_ins.tumor_mask = {};%储存肿瘤mask矩阵
    %如果包含其他模态，则提前构建容器，并置位多modeflag  {}{}{}{}
    if contain_orthermode
        subject_ins.tumor_othermode = {};
        subject_ins.tumor_mask_othermode = {};
    end
    
    
    
    %统计该病人所有肿瘤个数
    num_tumor = 0;
    for ii = 1:length(subject_ins.v_m_id)
        num_tumor=num_tumor+length(subject_ins.v_m_id{ii});
    end
    disp(['tumors num :',num2str(num_tumor)]);
    
    subject_ins.num_tumor = num_tumor;
    
    
    tumor_index = 0;
    %提取CT
    for ii = 1:length(subject_ins.v_id)
        sec_id = subject_ins.v_id(ii);
        CT_file_name =strcat('s',num2str(subject_ins.id),'_','v',num2str(sec_id),'.nii');
        Vref = spm_vol(strcat(nii_path,filesep,CT_file_name));
        Data_CT = spm_read_vols(Vref);%加载完成CT图像，接下来加载CT并抠出
        tmp_mask_id_list = subject_ins.v_m_id{ii};%提取对应肿瘤的mask id 矩阵
        
        %如果包括其他mode，则把其他mode的CT也提取出来
        if contain_orthermode
            Vref = spm_vol(strcat(nii_othermode_path,filesep,CT_file_name));
            Data_CT_othermode = spm_read_vols(Vref);%加载完成CT图像，接下来加载CT并抠出
        end
        
        %保存每个CT的体素size
        voxel_size = abs([Vref.mat(1,1),Vref.mat(2,2),Vref.mat(3,3)]);
        subject_ins.voxel_size{end+1} = voxel_size;
        
        %预构建储存tumorsize以及location的元祖，树状图储存（也就是分CT号储存，而不是像之前那样储存）
        subject_ins.tumor_shape{ii}={};
        subject_ins.tumor_location{ii}={};
        
        
        %预构建储存tumor的元祖
        subject_ins.tumor{ii} = {};
        subject_ins.tumor_mask{ii} = {};
        if contain_orthermode
            subject_ins.tumor_othermode{ii} = {};
            subject_ins.tumor_mask_othermode{ii} = {};
        end
        
        
        %在对应肿瘤的mask id 矩阵中循环，抠出，预处理，并保存到结构体
        for iii = 1:length(tmp_mask_id_list)
            tumor_index = tumor_index+1;
            disp(['loading tumor',num2str(tumor_index),'...']);
            trd_id = tmp_mask_id_list(iii);
            
            % 读取肿瘤mask
            mask_file_name =strcat('s',num2str(subject_ins.id),'_','v',num2str(sec_id),'_m',num2str(trd_id),'.nii');
            Vref = spm_vol(strcat(nii_path,filesep,mask_file_name));
            Data_mask = spm_read_vols(Vref);
            
            
            
            
            %如果包括其他mode，则把其他mode的mask也提取出来
            if contain_orthermode
                Vref = spm_vol(strcat(nii_othermode_path,filesep,mask_file_name));
                Data_mask_othermode = spm_read_vols(Vref);%加载完成CT图像，接下来加载CT并抠出
            end
            
            % 数据处理流现在开始 ============================
            % ==============================================
            % ==============================================
            % ==============================================
            % ==============================================
            % flow 1:使用抠出函数，基于mask将肿瘤从CT中抠出
            [tumor_mat, mask_mat, tumor_size,or_shape,or_location] = get_region_from_mask(Data_CT, Data_mask, extend_pixel);%参数：前者CT，后者mask
            if contain_orthermode
                [tumor_mat_other, mask_mat_other,~,~,~] = get_region_from_mask(Data_CT_othermode, Data_mask_othermode, extend_pixel);
                %因为动脉静脉期roi不一样，所以需要将二者resize为一样大小
                %如果以后有配准的话，可能先配准，这样这里就不需要了（当然也可以直接在这里配准）
                tumor_mat_other = imresize3(tumor_mat_other,or_shape,'cubic');
                mask_mat_other = imresize3(mask_mat_other,or_shape,'cubic');
            end
            
            % flow 2:预处理：调整窗宽窗位
            tumor_mat = CT_data_preprocess(tumor_mat,'window_change',adjust_window);
            if contain_orthermode
                % 静脉期可能需要不同的窗宽窗位：:9 220 [200 -20]
                tumor_mat_other = CT_data_preprocess(tumor_mat_other,'window_change',adjust_window4_othermode);
            end
            
            % flow 3:预处理：空间体素重采样
            if resample_flag
                tumor_mat = CT_data_preprocess(tumor_mat,'voxel_dim_resampling',voxel_size,adjust_voxelsize);
                mask_mat = CT_data_preprocess(mask_mat,'voxel_dim_resampling',voxel_size,adjust_voxelsize);
                if contain_orthermode
                    % 静脉期可能需要不同的窗宽窗位：:9 220 [200 -20]
                    tumor_mat_other = CT_data_preprocess(tumor_mat_other,'voxel_dim_resampling',voxel_size,adjust_voxelsize);
                    mask_mat_other = CT_data_preprocess(mask_mat_other,'voxel_dim_resampling',voxel_size,adjust_voxelsize);
                end
            end
            
            
            
            % flow 4:预处理：线性归一化
            tumor_mat = CT_data_preprocess(tumor_mat,'Linear_normalization');
            if contain_orthermode
                tumor_mat_other = CT_data_preprocess(tumor_mat_other,'Linear_normalization');
            end
            % 数据处理流现在结束 ============================
            % ==============================================
            % ==============================================
            % ==============================================
            % ==============================================

            
            
            %保存到结构体
            subject_ins.tumor{ii}{end+1}=tumor_mat;
            subject_ins.tumor_mask{ii}{end+1}=mask_mat;
            subject_ins.tumor_shape{ii}{end+1}=or_shape;
            subject_ins.tumor_location{ii}{end+1}=or_location;
%             subject_ins.tumor_shape{end+1}=or_shape;
            subject_ins.tumor_size(end+1)=tumor_size;%体素数量，不是真正体积
            
            if contain_orthermode
                subject_ins.tumor_othermode{ii}{end+1}=tumor_mat_other;
                subject_ins.tumor_mask_othermode{ii}{end+1}=mask_mat_other;
            end
            

        end
%        subject(i).tumor_shape{ii} = subject_ins.tumor_shape;
        
    end
    % 当一个病人全部肿瘤都抠完之后，储存该病人所有肿瘤个数
    subject_ins.tumor_size_all = sum(subject_ins.tumor_size);
    % 保存结构体subject_temp到指定文件夹
    disp('saving...');
    save(strcat(save_path,filesep,num2str(subject_ins.id)),'subject_ins');   
    
    % 顺便把数据也保存到汇总用的那个结构体数组中
    subject(i).num_tumor = subject_ins.num_tumor;%将肿瘤个数保存到结构体数组中
    subject(i).tumor_size = subject_ins.tumor_size;
    subject(i).tumor_size_all = subject_ins.tumor_size_all;
    subject(i).tumor_shape = subject_ins.tumor_shape;
    subject(i).voxel_size = subject_ins.voxel_size;
    subject(i).tumor_location = subject_ins.tumor_location;
end

mkdir(strcat(save_path,filesep,'subject_all'));
save(strcat(save_path,filesep,'subject_all',filesep,'subject'),'subject',...
    'adjust_window','adjust_window4_othermode','extend_pixel','adjust_voxelsize','resample_flag'); 















