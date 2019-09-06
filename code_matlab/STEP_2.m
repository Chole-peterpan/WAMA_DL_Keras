%% 完成分块操作，并将分块信息整合到subject结构体数组储存
%% 初始化
clc;
clear;

%% 设置参数
mat_path =       'H:\@data_NENs_recurrence\PNENs\data_outside\@flow3\1mat';
block_savepath = 'H:\@data_NENs_recurrence\PNENs\data_outside\@flow3\2block';
step = 5; % 分块的滑动步长
deepth = 20;% 分块的层厚
%% 读取结构体数组信息文件
subject_path = strcat(mat_path,filesep,'subject_all',filesep,'subject.mat');
workspaces = load(subject_path);
subject = workspaces.subject;%记录所有统计信息的mat文件，为结构体数组。

%% 对每个病人的每个肿瘤分块
for i = 1:length(subject)
    disp(['blocking subject',num2str(subject(i).id),'...']);
    filename = strcat(mat_path,filesep,num2str(subject(i).id),'.mat');
    
    workspaces = load(filename);
    subject_ins = workspaces.subject_ins;
    subject(i).blocks_num_per_tumor = [];%创建空矩阵储存每个肿瘤扩增的数量
   
    block_id = 0;%以病人为单位,计数该病人所有肿瘤一共分了多少块
    
    for ii = 1:subject_ins.num_tumor
        tumor_block_num = 0;%以肿瘤为单位
        tumor = subject_ins.tumor{ii};%获取第ii个肿瘤
        tumor_mask = subject_ins.tumor_mask{ii};%获取第ii个肿瘤对应的mask
        
        % 另一模态或者序列
        if subject_ins.othermode
            tumor_othermode = subject_ins.tumor_othermode{ii};%获取第ii个肿瘤
            tumor_mask_othermode = subject_ins.tumor_mask_othermode{ii};%获取第ii个肿瘤对应的mask
        end
        
        %执行分块操作
        [blocks_all,masks_all,~] = get_block_from_tumor(tumor,tumor_mask,step,deepth);%返回block的元组数组
        if subject_ins.othermode
            [blocks_all_othermode,masks_all_othermode,~] = get_block_from_tumor(tumor_othermode,tumor_mask_othermode,step,deepth);%返回block的元组数组
        end
        
        
        %保存分块，并将分块信息保存到subject_all以供日后统计。
        for iii = 1:length(blocks_all)
            block = blocks_all{iii};
            mask = masks_all{iii};
            
            if subject_ins.othermode
                block_othermode = blocks_all_othermode{iii};
                mask_othermode = masks_all_othermode{iii};
            end
            
            tumor_block_num = tumor_block_num+1;
            block_id = block_id+1;
            
            savepath = strcat(block_savepath,filesep,num2str(subject(i).id),'_',num2str(block_id));
            
            % 如果是多模态，则需要把多模态的也保存下来
            if subject_ins.othermode
                save(savepath,'block','mask','block_othermode','mask_othermode');
            else
                save(savepath,'block','mask'); 
            end
            

        end
        subject(i).blocks_num_per_tumor(end+1)=tumor_block_num;
 
    end
    subject(i).blocks_num_all = block_id;   
end 

mkdir(strcat(block_savepath,filesep,'subject'));
save(strcat(block_savepath,filesep,'subject',filesep,'subject.mat'),'subject','step','deepth'); 

















