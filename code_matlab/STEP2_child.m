%% ��ȡ�ṹ��������Ϣ�ļ�
subject_path = strcat(mat_path,filesep,'subject_all',filesep,'subject.mat');
workspaces = load(subject_path);
subject = workspaces.subject;%��¼����ͳ����Ϣ��mat�ļ���Ϊ�ṹ�����顣
% ������step1�̳й����ı���
data_path = workspaces.data_path;
data_window = workspaces.data_window;
xls_data = workspaces.xls_data;
resample_flag = workspaces.resample_flag;
extend_length = workspaces.extend_length;
adjust_voxelsize = workspaces.adjust_voxelsize;

%% ��ÿ�����˵�ÿ�������ֿ�
% ����ÿһ������
for i = 1:length(subject)
    frt_id = subject(i).id;
    disp(['blocking subject',num2str(subject(i).id),'...']);
    filename = strcat(mat_path,filesep,num2str(subject(i).id),'.mat');
    
    workspaces = load(filename);
    subject_ins = workspaces.subject_ins; 
    label = subject_ins.or_label;
    block_num = 0;%�Բ���Ϊ��λ,�����ò�����������һ�����˶��ٿ�
    
    % ��������ÿ��ɨ��� Ҳ����vid
    for ii = 1:length(subject_ins.v_id)
       scd_id = subject_ins.v_id(ii);
       subject_ins.blocks_num_per_tumor{ii}=[];
       % ����ÿһ������ÿ��ɨ��ε�����
       for iii = 1:length(subject_ins.v_m_id{ii})
           trd_id = subject_ins.v_m_id{ii}(iii);
           tumor_block_num = 0;%������Ϊ��λ,����������һ�����˶��ٿ�
           
           % ������ģ̬�������Ͷ�Ӧ��mask�ó���
           tumor = {};
           for n = 1:subject_ins.othermode
              tumor{end+1} =  subject_ins.tumor{n}{ii}{iii};
           end
           tumor_mask = {};
           for n = 1:subject_ins.othermode
              tumor_mask{end+1} =  subject_ins.tumor_mask{n}{ii}{iii};
           end
           
           %ִ�зֿ����
           [blocks_all,masks_all,~] = get_block(tumor,tumor_mask,step,deepth);%����block��Ԫ������

           %����ֿ飬�����ֿ���Ϣ���浽subject_all�Թ��պ�ͳ�ơ�
           for iiii = 1:length(blocks_all{1})
               fth_id = iiii;
               
               % ��ͬһblock������ģ̬ȡ����
               block = {};
               mask = {};
               for n = 1:length(blocks_all)
                   block{end+1} = blocks_all{n}{iiii};
                   mask{end+1} = masks_all{n}{iiii};
               end

               tumor_block_num = tumor_block_num+1;
               block_num = block_num+1;
               
               % ����
               savepath = strcat(block_savepath,filesep,'s',num2str(frt_id),'_v',num2str(scd_id),'_m',num2str(trd_id),'_b',num2str(fth_id));
               save(savepath,'block','mask','label');
 
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
save(strcat(block_savepath,filesep,'subject',filesep,'subject.mat'),'subject','step','deepth',...
    'data_path','data_window','xls_data','resample_flag','extend_length','adjust_voxelsize'); 

