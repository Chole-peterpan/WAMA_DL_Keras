%% ��ɷֿ�����������ֿ���Ϣ���ϵ�subject�ṹ�����鴢��
%% ��ʼ��
clc;
clear;

%% ���ò���
mat_path =       'H:\@data_NENs_recurrence\PNENs\data_outside\@flow3\1mat';
block_savepath = 'H:\@data_NENs_recurrence\PNENs\data_outside\@flow3\2block';
step = 5; % �ֿ�Ļ�������
deepth = 20;% �ֿ�Ĳ��
%% ��ȡ�ṹ��������Ϣ�ļ�
subject_path = strcat(mat_path,filesep,'subject_all',filesep,'subject.mat');
workspaces = load(subject_path);
subject = workspaces.subject;%��¼����ͳ����Ϣ��mat�ļ���Ϊ�ṹ�����顣

%% ��ÿ�����˵�ÿ�������ֿ�
for i = 1:length(subject)
    disp(['blocking subject',num2str(subject(i).id),'...']);
    filename = strcat(mat_path,filesep,num2str(subject(i).id),'.mat');
    
    workspaces = load(filename);
    subject_ins = workspaces.subject_ins;
    subject(i).blocks_num_per_tumor = [];%�����վ��󴢴�ÿ����������������
   
    block_id = 0;%�Բ���Ϊ��λ,�����ò�����������һ�����˶��ٿ�
    
    for ii = 1:subject_ins.num_tumor
        tumor_block_num = 0;%������Ϊ��λ
        tumor = subject_ins.tumor{ii};%��ȡ��ii������
        tumor_mask = subject_ins.tumor_mask{ii};%��ȡ��ii��������Ӧ��mask
        
        % ��һģ̬��������
        if subject_ins.othermode
            tumor_othermode = subject_ins.tumor_othermode{ii};%��ȡ��ii������
            tumor_mask_othermode = subject_ins.tumor_mask_othermode{ii};%��ȡ��ii��������Ӧ��mask
        end
        
        %ִ�зֿ����
        [blocks_all,masks_all,~] = get_block_from_tumor(tumor,tumor_mask,step,deepth);%����block��Ԫ������
        if subject_ins.othermode
            [blocks_all_othermode,masks_all_othermode,~] = get_block_from_tumor(tumor_othermode,tumor_mask_othermode,step,deepth);%����block��Ԫ������
        end
        
        
        %����ֿ飬�����ֿ���Ϣ���浽subject_all�Թ��պ�ͳ�ơ�
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
            
            % ����Ƕ�ģ̬������Ҫ�Ѷ�ģ̬��Ҳ��������
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

















