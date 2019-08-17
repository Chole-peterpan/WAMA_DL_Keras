%完成分块操作，并将分块信息整合到subject结构体数组储存
%初始化
clc;
clear;
mat_path = 'G:\@data_NENs_response\data_2D\1mat';
block_savepath = 'G:\@data_NENs_response\data_2D\2block';


subject_all_path = strcat(mat_path,filesep,'subject_all',filesep,'subject.mat');
workspaces = load(subject_all_path);
subject_all = workspaces.subject;%记录所有统计信息的mat文件，为结构体数组。

for i = 1:length(subject_all)
    disp(['blocking subject',num2str(subject_all(i).id),'...']);
    filename = strcat(mat_path,filesep,num2str(subject_all(i).id),'.mat');
    %filename = subject_list(i,1).name;
    workspaces = load(filename);
    subject = workspaces.subject_temp;
    subject_all(i).blocks_num_per_tumor = [];%创建空矩阵储存每个肿瘤扩增的数量
   
    block_id = 0;%以病人为单位
    
    for ii = 1:subject.v_num_tumor
        tumor_block_num = 0;%以肿瘤为单位
        tumor = subject.v_tumor{ii};%获取第ii个肿瘤
        tumor_mask = subject.v_tumor_mask{ii};%获取第ii个肿瘤对应的mask
        
        %执行分块操作
        [blocks_all,masks_all,~] = get_block_from_tumor(tumor,tumor_mask);%返回block的元组数组
        
        %保存分块，并将分块信息保存到subject_all以供日后统计。
        for iii = 1:length(blocks_all)
            block = blocks_all{iii};
            mask = masks_all{iii};
            
            tumor_block_num = tumor_block_num+1;
            block_id = block_id+1;
            
            savepath = strcat(block_savepath,filesep,num2str(subject_all(i).id),'_',num2str(block_id));
            save(savepath,'block','mask'); 
            
        end
        subject_all(i).blocks_num_per_tumor(end+1)=tumor_block_num;
        
        
    end
    subject_all(i).blocks_num_all = block_id;
      
end 

mkdir(strcat(block_savepath,filesep,'subject_all'));
save(strcat(block_savepath,filesep,'subject_all',filesep,'subject.mat'),'subject_all'); 

















