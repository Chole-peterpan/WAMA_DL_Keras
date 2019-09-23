%% ��ɷֿ�����������ֿ���Ϣ���ϵ�subject�ṹ�����鴢��
%% ��ʼ��
clc;
clear;

%% ���ò���
mat_path =       'G:\test\1mat';
block_savepath = 'G:\test\2block';
step = 5; % �ֿ�Ļ�������
deepth = 20;% �ֿ�Ĳ��
% ������������ע�⣬�������20���أ���ô�Ͳ�ҪdeepthΪ20��������û����������
%% ��ȡ�ṹ��������Ϣ�ļ�
subject_path = strcat(mat_path,filesep,'subject_all',filesep,'subject.mat');
workspaces = load(subject_path);
subject = workspaces.subject;%��¼����ͳ����Ϣ��mat�ļ���Ϊ�ṹ�����顣

%% ��ÿ�����˵�ÿ�������ֿ�
% ����ÿһ������
for i = 1:length(subject)
    frst_id = subject(i).id;
    disp(['blocking subject',num2str(subject(i).id),'...']);
    filename = strcat(mat_path,filesep,num2str(subject(i).id),'.mat');
    
    workspaces = load(filename);
    subject_ins = workspaces.subject_ins;  
    block_num = 0;%�Բ���Ϊ��λ,�����ò�����������һ�����˶��ٿ�
    
    % ��������ÿ��ɨ��� Ҳ����vid
    for ii = 1:length(subject_ins.v_id)
       scd_id = subject_ins.v_id(ii);
       subject_ins.blocks_num_per_tumor{ii}=[];
       % ����ÿһ������ÿ��ɨ��ε�����
       for iii = 1:length(subject_ins.v_m_id{ii})
           trd_id = subject_ins.v_m_id{ii}(iii);
           tumor_block_num = 0;%������Ϊ��λ,����������һ�����˶��ٿ�
           
           %��ȡ����
           tumor = subject_ins.tumor{ii}{iii};%��ȡ��ii��ɨ��εĵ�iii������
           tumor_mask = subject_ins.tumor_mask{ii}{iii};
           if subject_ins.othermode
               tumor_othermode = subject_ins.tumor_othermode{ii}{iii};
               tumor_mask_othermode = subject_ins.tumor_mask_othermode{ii}{iii};
           end
           
           %ִ�зֿ����
           [blocks_all,masks_all,~] = get_block_from_tumor(tumor,tumor_mask,step,deepth);%����block��Ԫ������
           if subject_ins.othermode
               [blocks_all_othermode,masks_all_othermode,~] = get_block_from_tumor(tumor_othermode,tumor_mask_othermode,step,deepth);%����block��Ԫ������
           end
           
           
           %����ֿ飬�����ֿ���Ϣ���浽subject_all�Թ��պ�ͳ�ơ�
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
               
               % ����Ƕ�ģ̬������Ҫ�Ѷ�ģ̬��Ҳ��������
               if subject_ins.othermode
                   save(savepath,'block','mask','block_othermode','mask_othermode');
               else
                   save(savepath,'block','mask');
               end 
           end
           % ���������ķֿ������浽��Ӧvid�£���ʱvid��Ӧii��
           subject_ins.blocks_num_per_tumor{ii}(end+1)=tumor_block_num;
       end 
    end
    % һ�������Ѿ�������ϣ���ʱ�������˵�ȫ����Ϣ����
    subject(i).blocks_num_per_tumor=subject_ins.blocks_num_per_tumor;   
    subject(i).blocks_num_all = block_num;   
end 

mkdir(strcat(block_savepath,filesep,'subject'));
save(strcat(block_savepath,filesep,'subject',filesep,'subject.mat'),'subject','step','deepth'); 

















