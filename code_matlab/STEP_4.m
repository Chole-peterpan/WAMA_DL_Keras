%% ������ʵʱ���������ԾͲ���Ҫ�����������ô�鷳��ֱ��ָ����������������������Ӧ�Ĳ����ͺ�
% ������Ȼ����step3�����ϸ�֤�ֲ�������ȴ���������
%% ��ʼ��
clc;
clear;

%% ��������
block_mat_path =       'G:\test\2block';%block
augdict.mat_savepath = 'G:\test\4aug_h5';

%������ѭ����Ŀ�Ľ�����Ϊ������matlab�Ĵ����۵�����
%���ò���
for i = 1:1
    augdict.aug_num = 100;%����������
    augdict.class_a_id = 1:49;% �ֶ�����a�ಡ�˵�id
    augdict.class_b_id = 50:59;% �ֶ�����b�ಡ�˵�id
    augdict.a_b_ratio = [1,1];%������������a��b������AB����Ϊ1:2��������Ϊ[1,2],A�Ƿ�PD��B��PD
    
    % ��������ת
    augdict.rotation.flag = true;%�Ƿ�ʹ����ת����
    augdict.rotation.range = [0,180];%��ת�����ĽǶȷ�Χ
    augdict.rotation.p = 0.7;
    
    % �������Աȶȵ���
    augdict.gray_adjust.flag = true;%�Ƿ�ʹ�öԱȶȵ���
    augdict.gray_adjust.up = [0.9   ,   1];%�Աȶȵ������Ͻ緶Χ
    augdict.gray_adjust.low = [0    ,   0.1];%�Աȶȵ������½緶Χ
    augdict.gray_adjust.p = 0.7;
    
    % ���������ҷ�ת LEFT RIGHT
    % ���ڸ��ʣ�������ɾ��ȷֲ���0��1������ǰٷ�֮5���ʷ�ת���Ǿͳ���С��0.05�����־ͷ�ת
    augdict.LR_overturn.flag = true;%�Ƿ�ʹ�����ҷ�ת
    augdict.LR_overturn.p = 0.5;%���ҷ�ת�ĸ���
    
    
    % ���������·�ת UP DOWN
    augdict.UD_overturn.flag = true;%�Ƿ�ʹ�����·�ת
    augdict.UD_overturn.p = 0.5;%���·�ת�ĸ���
    
    % ������󷵻ص���ʽ���ƣ������Ǹ��ִ�С��Ҫ�󷵻ص�shape��ͬ
    % 0�����ı���״
    % mode 1 �� 3Dresize
    % mode 2 �� ��������
    augdict.savefomat.mode = 4;
    augdict.savefomat.param = [180,180,20];
    
    
    % ��������ӣ��Ա㸴��ʵ��
    augdict.s = rng;
end



%% ���ز�����Ϣ
workspaces = load(strcat(block_mat_path,filesep,'subject',filesep,'subject.mat'));
subject = workspaces.subject;%��¼����ͳ����Ϣ��mat�ļ���Ϊ�ṹ�����顣


sub_num_A = 0;%��PD����
sub_num_B = 0;%PD����
%�����˼��ϱ�ǩ=======================================================================
for i = 1:length(subject)
    %���������������򣬸���������Ӧ��label
    %����ǰ20�����������������Ǹ�������ʵ������
    if ismember(subject(i).id, augdict.class_a_id)
        label_1 = 0;label_2 = 0;label_3 = 1;%�ֶ�����
        sub_num_A = sub_num_A+1;
    elseif ismember(subject(i).id, augdict.class_b_id)
        label_1 = 0;label_2 = 1;label_3 = 0;%�ֶ�����
        sub_num_B = sub_num_B+1;
    end
    
    % ����3������ȱ���
    subject(i).label = [label_1;label_2;label_3];
    
    % ����2������ȱ���
    subject(i).label1 = [label_1;label_2];
    subject(i).label2 = [label_1;label_3];
    subject(i).label3 = [label_2;label_3];
    
    % ����2������ͨlabel
    subject(i).label_1 = label_1;
    subject(i).label_2 = label_2;
    subject(i).label_3 = label_3;
end

%����ÿһ������������������
aug_num_A = augdict.aug_num*(augdict.a_b_ratio(1)/sum(augdict.a_b_ratio));%A������������������
aug_num_B = augdict.aug_num*(augdict.a_b_ratio(2)/sum(augdict.a_b_ratio));%B������������������


%���������ÿ������������������
per_sub_augnum_A = (aug_num_A/sub_num_A);%A����ÿ�����������Ĵ������
per_sub_augnum_B = (aug_num_B/sub_num_B);%B����ÿ�����������Ĵ������


%% �������Բ���Ϊ��λ���в���������ÿ�����˵�ÿ��blockӦ�ñ��������ٿ鲢���浽�ṹ��������
for i = 1:length(subject)
    subject_ins = subject(i);
    blocks = subject_ins.blocks_num_all;
    % �ж���𲢼���ò��˵�ÿ��block����������
    if ismember(subject(i).id, augdict.class_a_id)
        per_sub_augnum = per_sub_augnum_A;
    elseif  ismember(subject(i).id, augdict.class_b_id)
        per_sub_augnum = per_sub_augnum_B;
    end
    per_block_augnum = ceil(per_sub_augnum/blocks);
    
    %����Ϣ���浽���˽ṹ��������
    subject(i).per_block_aug_num = per_block_augnum;
    subject(i).per_tumor_aug_num = {};
    all_aug_num = 0;
    for ii = 1:length(subject(i).blocks_num_per_tumor)
        subject(i).per_tumor_aug_num{ii}= per_block_augnum*(subject(i).blocks_num_per_tumor{ii});
        all_aug_num  = all_aug_num + sum(subject(i).per_tumor_aug_num{ii});
    end
    
    subject(i).all_aug_num = all_aug_num;

end


%% ֮��ѭ��block�б�ÿ��ȡһ��block�����ҵ�����id��Ӧ��per_block_augnum��֮�������������ˣ��ǵð�����ϸ�ڱ��浽block_detail
block_detail = [];
filename_list = dir(strcat(block_mat_path,filesep,'*.mat'));
for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    data_path=strcat(block_mat_path,filesep,filename);
    
    workspaces = load(data_path);
    data = workspaces.block;
    data = mat2gray(data);%һ��Ҫ��һ������Ȼimadjust��ʱ���ضϺܴ�һ����ֵ������������������
    if subject(i).othermode
        data_othermode = workspaces.block_othermode;
        data_othermode = mat2gray(data_othermode);%һ��Ҫ��һ������Ȼimadjust��ʱ���ضϺܴ�һ����ֵ������������������
    end
    

    tmpcell = strsplit(filename,'_');
    id = str2double(tmpcell{1}(2:end));
    for id_index = 1:length(subject)
       if id ==  subject(id_index).id
          per_block_augnum =  subject(id_index).per_block_aug_num;
       end
    end
    
    
    for iii = 1:per_block_augnum
        % ����
        if subject(i).othermode
            [aug_data,aug_data_othermode,aug_detail] = aug43D(data,  augdict,  subject(i).othermode,  data_othermode);
        else
            [aug_data,~                 ,aug_detail] = aug43D(data,  augdict,  subject(i).othermode,  []);
        end
              
        % ���� �� detail����
        tmp_name = strcat(filename(1:end-4),'_e',num2str(iii));
        block_detail(end+1).name = tmp_name;
        block_detail(end).aug_detail = aug_detail;
        
        
        write_name = strcat(tmp_name,'.h5');
        fprintf('aug_num:%d    aug_file_name: %s\n',iii,write_name);
        finalpath = strcat(augdict.mat_savepath,filesep,write_name);%���մ���·��
        disp(finalpath);
        % д��H5
        h5create(finalpath, '/data', size(aug_data),'Datatype','single');
        h5write(finalpath, '/data', aug_data);
        
        if subject(i).othermode
            h5create(finalpath, '/data_othermode', size(aug_data_othermode),'Datatype','single');
            h5write(finalpath, '/data_othermode', aug_data_othermode);
        end
        
        
        h5create(finalpath, '/label_1', size(subject(i).label_1),'Datatype','single');
        h5write(finalpath, '/label_1', subject(i).label_1);
        h5create(finalpath, '/label_2', size(subject(i).label_2),'Datatype','single');
        h5write(finalpath, '/label_2', subject(i).label_2);
        h5create(finalpath, '/label_3', size(subject(i).label_3),'Datatype','single');
        h5write(finalpath, '/label_3', subject(i).label_3);
        h5create(finalpath, '/label', size(subject(i).label),'Datatype','single');
        h5write(finalpath, '/label', subject(i).label);
        h5create(finalpath, '/label1', size(subject(i).label1),'Datatype','single');
        h5write(finalpath, '/label1', subject(i).label1);
        h5create(finalpath, '/label2', size(subject(i).label2),'Datatype','single');
        h5write(finalpath, '/label2', subject(i).label2);
        h5create(finalpath, '/label3', size(subject(i).label3),'Datatype','single');
        h5write(finalpath, '/label3', subject(i).label3);
        
        
    end
end
mkdir(strcat(augdict.mat_savepath,filesep,'subject'));
save(strcat(augdict.mat_savepath,filesep,'subject',filesep,'subject.mat'),'subject','augdict','block_detail');





































