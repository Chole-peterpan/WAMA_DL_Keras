%整理nii信息并将病人数据保存在特定结构体中，并分别保存病人数据到mat文件。
%同时储存所有病人信息以供日后统计
%初始化===============================================================
clc;
clear;
%设置参数
save_path = 'G:\@data_NENs_recurrence\PNENs\data\a\1mat';
nii_path = 'G:\@data_NENs_recurrence\PNENs\data\a\0nii';
%=====================================================================
%提取病例信息
nii_filename_list = dir(strcat(nii_path,filesep,'*.nii'));
filename_list = cell(length(nii_filename_list),1);
for i = 1:length(nii_filename_list)
    tmp_filename = nii_filename_list(i).name;
    filename_list{i}=tmp_filename(1:end-4);
end
subject = get_subject_from_filename(filename_list);

%由于我们目前只有静脉期，故只对静脉期进行操作,
%根据病例信息，对所有病例的mask抠出，并每一个病例单独保存一个文件，
%信息包括：subject结构体数组对应信息，肿瘤个数，对应肿瘤的体积，单个病人所有肿瘤的体积之和。
%这一步包括预处理，先扣出来再预处理（也可以之后尝试先预处理再扣出来）
sec_name = 'v';
for i = 1:length(subject)
    subject_temp =  subject(i);%这个就是最终保存的
    disp(['@ subject:',num2str(subject(i).id),'=====================']);
    
    %统计肿瘤个数
    num_tumor = 0;
    for ii = 1:length(subject_temp.v_m_id)
        num_tumor=num_tumor+length(subject_temp.v_m_id{ii});
    end
    subject(i).v_num_tumor = num_tumor;%将肿瘤个数保存到结构体数组中
    disp(['tumors num :',num2str(num_tumor)]);
    subject_temp.v_num_tumor = num_tumor;
    
    tumor_index = 0;
    %提取CT
    for ii = 1:length(subject(i).v_id)
        sec_id = subject(i).v_id(ii);
        CT_file_name =strcat('s',num2str(subject(i).id),'_',sec_name,num2str(sec_id),'.nii');
        Vref = spm_vol(strcat(nii_path,filesep,CT_file_name));
        Data_CT = spm_read_vols(Vref);%加载完成CT图像，接下来加载CT并抠出
        tmp_mat = subject_temp.v_m_id{ii};%提取对应肿瘤的mask id 矩阵
        
        %在对应肿瘤的mask id 矩阵中循环，抠出，预处理，并保存到结构体
        for iii = 1:length(tmp_mat)
            tumor_index = tumor_index+1;
            disp(['loading tumor',num2str(tumor_index)]);
            trd_id = tmp_mat(iii);
            mask_file_name =strcat('s',num2str(subject(i).id),'_',...
                sec_name,num2str(sec_id),'_m',num2str(trd_id),'.nii');
            
            Vref = spm_vol(strcat(nii_path,filesep,mask_file_name));
            Data_mask = spm_read_vols(Vref);%加载完成mask图像，接下来加载CT并抠出
            
            %使用抠出函数get_region_from_mask抠出对应区域,并返回体积
            [tumor_mat,mask_mat,tumor_size,~] = get_region_from_mask(Data_CT,Data_mask,5);%参数：前者CT，后者mask
            %预处理函数tumor_preprocess
            tumor_mat = tumor_preprocess(tumor_mat);
            %保存到结构体
            subject_temp.v_tumor{end+1}=tumor_mat;
            subject_temp.v_tumor_mask{end+1}=mask_mat;
            subject_temp.v_tumor_size(end+1)=tumor_size;%体素数量，不是真正体积
        end
        
    end
    subject_temp.v_tumor_size_all = sum(subject_temp.v_tumor_size);
    %保存结构体subject_temp到指定文件夹
    disp('saving...');
    save(strcat(save_path,filesep,num2str(subject_temp.id)),'subject_temp');   
    
    subject(i).v_tumor_size = subject_temp.v_tumor_size;
    subject(i).v_tumor_size_all = subject_temp.v_tumor_size_all;
    
    
    
end

mkdir(strcat(save_path,filesep,'subject_all'));
save(strcat(save_path,filesep,'subject_all',filesep,'subject'),'subject'); 















