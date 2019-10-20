%% 读取结构体数组信息文件
subject_path = strcat(mat_path,filesep,'subject_all',filesep,'subject.mat');
workspaces = load(subject_path);
subject = workspaces.subject;%记录所有统计信息的mat文件，为结构体数组。
% 其他从step1继承过来的变量
data_path = workspaces.data_path;
data_window = workspaces.data_window;
xls_data = workspaces.xls_data;
resample_flag = workspaces.resample_flag;
extend_length = workspaces.extend_length;
adjust_voxelsize = workspaces.adjust_voxelsize;

%% 对每个病人的每个肿瘤分块
% 遍历每一个病人
for i = 1:length(subject)
    frt_id = subject(i).id;
    disp(['blocking subject',num2str(subject(i).id),'...']);
    filename = strcat(mat_path,filesep,num2str(subject(i).id),'.mat');
    
    workspaces = load(filename);
    subject_ins = workspaces.subject_ins; 
    label = subject_ins.or_label;
    block_num = 0;%以病人为单位,计数该病人所有肿瘤一共分了多少块
    
    % 遍历病人每个扫描段 也就是vid
    for ii = 1:length(subject_ins.v_id)
       scd_id = subject_ins.v_id(ii);
       subject_ins.blocks_num_per_tumor{ii}=[];
       % 遍历每一个病人每个扫描段的肿瘤
       for iii = 1:length(subject_ins.v_m_id{ii})
           trd_id = subject_ins.v_m_id{ii}(iii);
           tumor_block_num = 0;%以肿瘤为单位,计数该肿瘤一共分了多少块
           
           % 将所有模态的肿瘤和对应的mask拿出来
           tumor = {};
           for n = 1:subject_ins.othermode
              tumor{end+1} =  subject_ins.tumor{n}{ii}{iii};
           end
           tumor_mask = {};
           for n = 1:subject_ins.othermode
              tumor_mask{end+1} =  subject_ins.tumor_mask{n}{ii}{iii};
           end
           
           %执行分块操作
           [blocks_all,masks_all,~] = get_block(tumor,tumor_mask,step,deepth);%返回block的元组数组

           %保存分块，并将分块信息保存到subject_all以供日后统计。
           for iiii = 1:length(blocks_all{1})
               fth_id = iiii;
               
               % 把同一block的所有模态取出来
               block = {};
               mask = {};
               for n = 1:length(blocks_all)
                   block{end+1} = blocks_all{n}{iiii};
                   mask{end+1} = masks_all{n}{iiii};
               end

               tumor_block_num = tumor_block_num+1;
               block_num = block_num+1;
               
               % 保存
               savepath = strcat(block_savepath,filesep,'s',num2str(frt_id),'_v',num2str(scd_id),'_m',num2str(trd_id),'_b',num2str(fth_id));
               save(savepath,'block','mask','label');
 
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
save(strcat(block_savepath,filesep,'subject',filesep,'subject.mat'),'subject','step','deepth',...
    'data_path','data_window','xls_data','resample_flag','extend_length','adjust_voxelsize'); 

