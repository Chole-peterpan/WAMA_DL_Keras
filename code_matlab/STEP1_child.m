%% 根据文件名整理数据信息
nii_filename_list = dir(strcat(data_path{1},filesep,'*.nii'));
filename_list = cell(length(nii_filename_list),1);
for i = 1:length(nii_filename_list)
    tmp_filename = nii_filename_list(i).name;
    filename_list{i}=tmp_filename(1:end-4);
end
subject = get_subject_from_filename(filename_list, data_path);

%% 赋予病人label
xls_data = xlsread(label_file, 1);
for i = 1:length(subject)
    subject(i).or_label = xls_data((xls_data(:,1)==subject(i).id),2:end);
end

%% 逐个处理病人nii文件
%根据病例信息，将所有病例的mask或roi抠出，并每一个病例单独保存一个文件，
%信息包括：subject结构体数组对应信息，肿瘤个数，对应肿瘤的体积，单个病人所有肿瘤的体积之和。
%请注意查看数据处理流!!!!!!!
for i = 1:length(subject) % i 是病人的索引
    subject_ins =  subject(i);%单个病人的结构体，这个就是最终保存的单个病人mat文件
    
    % 显示进程
    if rem(i,2)==0
        disp(['/(@.@)/ doing with subject:',num2str(subject_ins.id),' ...']);
    else
        disp(['\(@.@)\ doing with subject:',num2str(subject_ins.id),' ...']);
    end
    
    % 汇总文件不需要储存肿瘤，单个文件subject_ins才需要，所以需要新建
    subject_ins.tumor = {};%储存肿瘤CT矩阵
    subject_ins.tumor_mask = {};%储存肿瘤mask矩阵
    
    %统计该病人所有肿瘤个数
    num_tumor = 0;
    for ii = 1:length(subject_ins.v_m_id)
        num_tumor=num_tumor+length(subject_ins.v_m_id{ii});
    end
    disp(['tumors num :',num2str(num_tumor)]);
    subject_ins.num_tumor = num_tumor;
    
    % 遍历每一个模态
    for ii = 1:subject_ins.othermode  % ii 是模态的索引
        disp(['processing mode ',num2str(ii)]);
        % 每一个模态单独计数总肿瘤数量
        tumor_index = 0;
        % 构建一级目录cell:模态级别
        subject_ins.tumor{end+1} = {};
        subject_ins.tumor_mask{end+1} = {};
        subject_ins.voxel_size{end+1} = {};
        subject_ins.tumor_shape{end+1} = {};
        subject_ins.tumor_location{end+1} = {};
        subject_ins.tumor_size{end+1} = {};
        
        % 遍历每一个/段CT或MRI扫描段
        for iii = 1:length(subject_ins.v_id) % iii 是扫描段的索引
            % 构建二级目录cell：CT分段级别
            subject_ins.tumor{end}{end+1} = {};
            subject_ins.tumor_mask{end}{end+1} = {};
            % subject_ins.voxel_size{end}{end+1} = {};% 这个到CT级别就够了，所以不需要三级cell目录了
            subject_ins.tumor_shape{end}{end+1} = {};
            subject_ins.tumor_location{end}{end+1} = {};
            subject_ins.tumor_size{end}{end+1} = {};
                        
            sec_id = subject_ins.v_id(iii);% 获取分段CT的id
            % 读取CT
            CT_file_name =strcat('s',num2str(subject_ins.id),'_v',num2str(sec_id),'.nii');
            Vref = spm_vol(strcat(data_path{ii},filesep,CT_file_name));
            Data_CT = spm_read_vols(Vref);%加载完成CT图像，接下来加载CT并抠出
            %保存每个CT的体素size
            voxel_size = abs([Vref.mat(1,1),Vref.mat(2,2),Vref.mat(3,3)]);
            subject_ins.voxel_size{end}{end+1} = voxel_size;
            
            
            % 遍历每一个肿瘤
            tmp_mask_id_list = subject_ins.v_m_id{iii};%提取对应肿瘤的mask id列表
            for iiii = 1:length(tmp_mask_id_list) % iiii 是肿瘤的索引
                tumor_index = tumor_index+1;
                disp(['loading mode :',num2str(ii),', tumor:',num2str(tumor_index),'...']);
                trd_id = tmp_mask_id_list(iiii);
                
                % 读取肿瘤mask
                mask_file_name =strcat('s',num2str(subject_ins.id),'_v',num2str(sec_id),'_m',num2str(trd_id),'.nii');
                Vref = spm_vol(strcat(data_path{ii},filesep,mask_file_name));
                Data_mask = spm_read_vols(Vref);
                
                

                
                % 数据处理流现在开始 ============================
                % ==============================================
                % ==============================================
                % ==============================================
                % ==============================================
                % flow 1:使用抠出函数，基于mask将肿瘤从CT中抠出
                extend_pixel = floor(extend_length / voxel_size(1));
                [tumor_mat, mask_mat, tumor_size,or_shape,or_location] = get_region_from_mask(Data_CT, Data_mask, extend_pixel);%参数：前者CT，后者mask
              
                
                % 3Dresize，默认resize到第一个模态的肿瘤大小
                if ii ~= 1
                    tumor_mat = imresize3(tumor_mat,subject_ins.tumor_shape{1}{iii}{iiii},'cubic');
                    mask_mat = imresize3(mask_mat,or_shape,'cubic');
                end
                                
                % flow 2:预处理：调整窗宽窗位
                tumor_mat = CT_data_preprocess(tumor_mat,'window_change',data_window{ii});
                
                % flow 3:预处理：空间体素重采样
                if resample_flag
                    tumor_mat = CT_data_preprocess(tumor_mat, 'voxel_dim_resampling', voxel_size, adjust_voxelsize);
                    mask_mat = CT_data_preprocess(mask_mat, 'voxel_dim_resampling', voxel_size, adjust_voxelsize);
                end

                % flow 4:预处理：线性归一化
                tumor_mat = CT_data_preprocess(tumor_mat,'Linear_normalization');
                % 数据处理流现在结束 ============================
                % ==============================================
                % ==============================================
                % ==============================================
                % ==============================================
                
                %保存各个指标到结构体
                subject_ins.tumor{end}{end}{end+1}=tumor_mat;
                subject_ins.tumor_mask{end}{end}{end+1}=mask_mat;
                subject_ins.tumor_shape{end}{end}{end+1}=or_shape;
                subject_ins.tumor_location{end}{end}{end+1}=or_location;
                subject_ins.tumor_size{end}{end}{end+1}=tumor_size;%体素数量，不是真正体积 
            end   
        end

    end
    % 计算总肿瘤体积，默认计算第一个模态的体积
    tumor_size_all = 0;
    for n = 1:length(subject_ins.tumor_size{1})
        for nn = 1:length(subject_ins.tumor_size{1}{n})
            tumor_size_all = tumor_size_all + (subject_ins.tumor_size{1}{n}{nn});
        end
    end
    subject_ins.tumor_size_all = tumor_size_all;
    % 顺便把数据也保存到汇总用的那个结构体数组中
    subject(i).num_tumor = subject_ins.num_tumor;
    subject(i).tumor_size = subject_ins.tumor_size;
    subject(i).tumor_size_all = subject_ins.tumor_size_all;
    subject(i).tumor_shape = subject_ins.tumor_shape;
    subject(i).voxel_size = subject_ins.voxel_size;
    subject(i).tumor_location = subject_ins.tumor_location;
    

    % 保存结构体subject_temp到指定文件夹
    disp('saving...');
    save(strcat(save_path,filesep,num2str(subject_ins.id)),'subject_ins');   

end

mkdir(strcat(save_path,filesep,'subject_all'));
save(strcat(save_path,filesep,'subject_all',filesep,'subject'),'subject',...
    'data_path','data_window','xls_data','resample_flag','extend_length','adjust_voxelsize'); 


