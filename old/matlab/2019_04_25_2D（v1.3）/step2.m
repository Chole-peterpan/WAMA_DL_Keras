%��ɷֿ�����������ֿ���Ϣ���ϵ�subject�ṹ�����鴢��
%��ʼ��
clc;
clear;
mat_path = 'G:\@data_NENs_response\data_2D\1mat';
block_savepath = 'G:\@data_NENs_response\data_2D\2block';


subject_all_path = strcat(mat_path,filesep,'subject_all',filesep,'subject.mat');
workspaces = load(subject_all_path);
subject_all = workspaces.subject;%��¼����ͳ����Ϣ��mat�ļ���Ϊ�ṹ�����顣

for i = 1:length(subject_all)
    disp(['blocking subject',num2str(subject_all(i).id),'...']);
    filename = strcat(mat_path,filesep,num2str(subject_all(i).id),'.mat');
    %filename = subject_list(i,1).name;
    workspaces = load(filename);
    subject = workspaces.subject_temp;
    subject_all(i).blocks_num_per_tumor = [];%�����վ��󴢴�ÿ����������������
   
    block_id = 0;%�Բ���Ϊ��λ
    
    for ii = 1:subject.v_num_tumor
        tumor_block_num = 0;%������Ϊ��λ
        tumor = subject.v_tumor{ii};%��ȡ��ii������
        tumor_mask = subject.v_tumor_mask{ii};%��ȡ��ii��������Ӧ��mask
        
        %ִ�зֿ����
        [blocks_all,masks_all,~] = get_block_from_tumor(tumor,tumor_mask);%����block��Ԫ������
        
        %����ֿ飬�����ֿ���Ϣ���浽subject_all�Թ��պ�ͳ�ơ�
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

















