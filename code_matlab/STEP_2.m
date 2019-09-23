%% 完成分块操作，并将分块信息整合到subject结构体数组储存
%% 初始化
clc;
clear;

%% 设置参数
mat_path =       'G:\test\1mat';
block_savepath = 'G:\test\2block';
step = 5; % 分块的滑动步长
deepth = 20;% 分块的层厚
% 上面两个参数注意，如果外扩20体素，那么就不要deepth为20，这样就没多少肿瘤了
%% 读取结构体数组信息文件
subject_path = strcat(mat_path,filesep,'subject_all',filesep,'subject.mat');
workspaces = load(subject_path);
subject = workspaces.subject;%记录所有统计信息的mat文件，为结构体数组。

%% 对每个病人的每个肿瘤分块
% 遍历每一个病人
for i = 1:length(subject)
    frst_id = subject(i).id;
    disp(['blocking subject',num2str(subject(i).id),'...']);
    filename = strcat(mat_path,filesep,num2str(subject(i).id),'.mat');
    
    workspaces = load(filename);
    subject_ins = workspaces.subject_ins;  
    block_num = 0;%以病人为单位,计数该病人所有肿瘤一共分了多少块
    
    % 遍历病人每个扫描段 也就是vid
    for ii = 1:length(subject_ins.v_id)
       scd_id = subject_ins.v_id(ii);
       subject_ins.blocks_num_per_tumor{ii}=[];
       % 遍历每一个病人每个扫描段的肿瘤
       for iii = 1:length(subject_ins.v_m_id{ii})
           trd_id = subject_ins.v_m_id{ii}(iii);
           tumor_block_num = 0;%以肿瘤为单位,计数该肿瘤一共分了多少块
           
           %读取肿瘤
           tumor = subject_ins.tumor{ii}{iii};%获取第ii个扫描段的第iii个肿瘤
           tumor_mask = subject_ins.tumor_mask{ii}{iii};
           if subject_ins.othermode
               tumor_othermode = subject_ins.tumor_othermode{ii}{iii};
               tumor_mask_othermode = subject_ins.tumor_mask_othermode{ii}{iii};
           end
           
           %执行分块操作
           [blocks_all,masks_all,~] = get_block_from_tumor(tumor,tumor_mask,step,deepth);%返回block的元组数组
           if subject_ins.othermode
               [blocks_all_othermode,masks_all_othermode,~] = get_block_from_tumor(tumor_othermode,tumor_mask_othermode,step,deepth);%返回block的元组数组
           end
           
           
           %保存分块，并将分块信息保存到subject_all以供日后统计。
           for iiii = 1:length(blocks_all)
               fth_id = iiii;
               block = blocks_all{iiii};
               mask = masks_all{iiii};
               if subject_ins.othermode
                   block_othermode = blocks_all_othermode{iiii};
                   mask_othermode = masks_all_othermode{iiii};
               end
               
               tumor_block_num = tumor_block_num+1;
               block_num = block_num+1;
               
               savepath = strcat(block_savepath,filesep,'s',num2str(frst_id),'_v',num2str(scd_id),'_m',num2str(trd_id),'_b',num2str(fth_id));
               
               % 如果是多模态，则需要把多模态的也保存下来
               if subject_ins.othermode
                   save(savepath,'block','mask','block_othermode','mask_othermode');
               else
                   save(savepath,'block','mask');
               end 
           end
           % 将该肿瘤的分块数保存到对应vid下（此时vid对应ii）
           subject_ins.blocks_num_per_tumor{ii}(end+1)=tumor_block_num;
       end 
    end
    % 一个病人已经运行完毕，此时单个病人的全部信息储存
    subject(i).blocks_num_per_tumor=subject_ins.blocks_num_per_tumor;   
    subject(i).blocks_num_all = block_num;   
end 

mkdir(strcat(block_savepath,filesep,'subject'));
save(strcat(block_savepath,filesep,'subject',filesep,'subject.mat'),'subject','step','deepth'); 

















