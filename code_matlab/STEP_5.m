%% ������֤���۴��룬���ΪtxtĿ¼���������ڶ��������⣩
% ��������۲���Ϊ��ÿһ�൥������������֤ÿ����������ƽ��
% ���������������ݴ����Ĵ���
% ע����������ʱ��Ҫһ���ţ���Ҫ�������ȥ��

% �ļ��������򣺵�һλ����Ϊ������ţ��ڶ�λ����Ϊblock��ţ�����λ����Ϊ�������
% ��һ��������������blockĿ¼��augĿ¼��testĿ¼����cell��ע�⣬cut��Ҫ��������룬aug��block��������ֱ��ȫ������һ���ļ��м��ɣ�
% �ڶ������÷������ܵ�ʱ���������ݷŵ�һ���ļ��м��ɣ�ѵ����������aug���ݣ���֤������������Ҳ��aug��һ���ļ��У�test��δ�����ģ�
%%
clear
clc
%% ��������
K_fold = 8;
savepath = 'H:\@data_NENs_recurrence\PNENs\data\flow1\5CV';
%AUG data��h5�ļ���
aug_fold_cell{1}='H:\@data_NENs_recurrence\PNENs\data\flow1\4aug';
%or data��h5�ļ���
or_fold_cell{1}='H:\@data_NENs_recurrence\PNENs\data\flow1\3or';
%��������folder�ļ�����
have_folder = true;
folder_file = 'H:\@data_NENs_recurrence\PNENs\data\flow1\5CV\mat_folder.mat';
%% cut data Ϊ�˷��۵õ���ǩ��ǰ�������������۵Ľ��(mat�ļ�)
augdict.class_a_id = [[1:43],[46,47,49],[58,59]];% �ֶ�����a�ಡ�˵�id
augdict.class_b_id = 50:57;% �ֶ�����b�ಡ�˵�id
% augdict.class_a_id = 1:3;% �ֶ�����a�ಡ�˵�id
% augdict.class_b_id = 4:8;% �ֶ�����b�ಡ�˵�id

cut_n_name_final = num2cell(augdict.class_a_id);
cut_p_name_final = num2cell(augdict.class_b_id);
for i = 1:length(cut_n_name_final)
    cut_n_name_final{i} = num2str(cut_n_name_final{i});
end
for i = 1:length(cut_p_name_final)
    cut_p_name_final{i} = num2str(cut_p_name_final{i});
end
cut_n_name_final = cut_n_name_final';
cut_p_name_final = cut_p_name_final';


all_subject_name = [cut_n_name_final;cut_p_name_final];


%% ��ȡ�ļ���
aug_h5_filename = [];
or_h5_filename = [];


aug_filename_final = {};
test_filename_final = {};


%AUG----------------------------------------------------------------
for ii = 1:size(aug_fold_cell,2)
    tmp_filename = dir(strcat(aug_fold_cell{ii},filesep,'*.h5'));
    aug_h5_filename = [aug_h5_filename;tmp_filename];
end
for ii = 1:size(aug_h5_filename,1)
    aug_filename_final{ii}=aug_h5_filename(ii).name;    
end
aug_filename_final = aug_filename_final';

%TEST----------------------------------------------------------------
for ii = 1:size(or_fold_cell,2)
    tmp_filename = dir(strcat(or_fold_cell{ii},filesep,'*.h5'));
    or_h5_filename = [or_h5_filename;tmp_filename];
end
for ii = 1:size(or_h5_filename,1)
    test_filename_final{ii}=or_h5_filename(ii).name;    
end
test_filename_final = test_filename_final';


%% ����cut�����������ۣ�������Žṹ������
%����Ѿ��и����ͱ������·����ˣ�ֱ����֮ǰ��
if have_folder
    wkspace = load(folder_file);
    folder = wkspace.folder;
else
    
    %������֤�õ�index
    n_index = crossvalind('kfold',size(cut_n_name_final,1),K_fold);
    p_index = crossvalind('kfold',size(cut_p_name_final,1),K_fold);
    
    %����index����������ŷ���fold�ṹ��������test��
    for ii = 1:K_fold
        n_tmp_index = (n_index==ii);
        p_tmp_index = (p_index==ii);
        test_data_n = cut_n_name_final(n_tmp_index,:);
        test_data_p = cut_p_name_final(p_tmp_index,:);
        folder(ii).test = [test_data_n;test_data_p];
    end
    
    %�����ṹ����verify����
    for ii = 1:K_fold-1
        folder(ii).verify = folder(ii+1).test;
    end
    folder(K_fold).verify = folder(1).test;
    
    %�����ṹ����train����
    for ii = 1:K_fold
        tmp_file = {};
        for iii = 1:size(all_subject_name,1)
            if  ~ismember(all_subject_name{iii},folder(ii).test) && ~ismember(all_subject_name{iii},folder(ii).verify)
                tmp_file = [tmp_file;all_subject_name{iii}];
            end
        end
        folder(ii).train = tmp_file;
    end
    
end
%% ������Žṹ�����飬����ʵ���ļ�������
folder_final = [];
for iiii = 1:K_fold
    disp(strcat('making fold:',num2str(iiii),'...'));
    % ==========================================================
    % test������Ӧ��h5�ļ���
    or_test = {};
    for iii = 1:size(test_filename_final,1)
        tmp_str = test_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(2:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).test)
            or_test = [or_test;tmp_str];
        end 
    end
    folder_final(iiii).or_test = or_test;
    
    aug_test = {};
    for iii = 1:size(aug_filename_final,1)
        tmp_str = aug_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(2:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).test)
            aug_test = [aug_test;tmp_str];
        end
    end
    folder_final(iiii).aug_test = aug_test;
    % ==========================================================
    % ver������Ӧ��h5�ļ���
    or_verify = {};
    for iii = 1:size(test_filename_final,1)
        tmp_str = test_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(2:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).verify)
            or_verify = [or_verify;tmp_str];
        end 
    end
    folder_final(iiii).or_verify = or_verify; 
    
    aug_verify = {};
    for iii = 1:size(aug_filename_final,1)
        tmp_str = aug_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(2:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).verify)
            aug_verify = [aug_verify;tmp_str];
        end
    end
    folder_final(iiii).aug_verify = aug_verify;
    % ==========================================================
    % train������Ӧ��h5�ļ���
    or_train = {};
    for iii = 1:size(test_filename_final,1)
        tmp_str = test_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(2:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).train)
            or_train = [or_train;tmp_str];
        end
    end
    folder_final(iiii).or_train = or_train;
    
    aug_train = {};
    for iii = 1:size(aug_filename_final,1)
        tmp_str = aug_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(2:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).train)
            aug_train = [aug_train;tmp_str];
        end
    end
    folder_final(iiii).aug_train = aug_train;   
end


%% ���ΪTXT�Թ�pythonʹ��

for iii = 1:K_fold
    savename4_or_test = strcat(savepath,filesep,'fold_',num2str(iii),'_or_test.txt');
    savename4_aug_test = strcat(savepath,filesep,'fold_',num2str(iii),'_aug_test.txt');
    savename4_or_verify = strcat(savepath,filesep,'fold_',num2str(iii),'_or_verify.txt');
    savename4_aug_verify = strcat(savepath,filesep,'fold_',num2str(iii),'_aug_verify.txt');
    savename4_or_train = strcat(savepath,filesep,'fold_',num2str(iii),'_or_train.txt');
    savename4_aug_train = strcat(savepath,filesep,'fold_',num2str(iii),'_aug_train.txt');
  
    

    % save the test in TXT
	fid=fopen(savename4_or_test, 'w'); % �ļ�cΪ�������Ϻ���ļ�===============
    test = folder_final(iii).or_test;
    for n=1:length(test)
        fprintf(fid,'%s\n',char(test{n}));    %  
    end
    fclose(fid);
    disp('finish or_test filepath trans'); 
    
    fid=fopen(savename4_aug_test, 'w'); % �ļ�cΪ�������Ϻ���ļ�===============
    test = folder_final(iii).aug_test;
    for n=1:length(test)
        fprintf(fid,'%s\n',char(test{n}));    %
    end
    fclose(fid);
    disp('finish aug_test filepath trans');
    
    
    
    % save the verify in TXT
    fid=fopen(savename4_or_verify, 'w'); % �ļ�cΪ�������Ϻ���ļ�=============
    verify = folder_final(iii).or_verify;
    for n=1:length(verify)
        fprintf(fid,'%s\n',char(verify{n}));    %
    end
    fclose(fid);
    disp('finish or_verify filepath trans');
    
    fid=fopen(savename4_aug_verify, 'w'); % �ļ�cΪ�������Ϻ���ļ�=============
    verify = folder_final(iii).aug_verify;
    for n=1:length(verify)
        fprintf(fid,'%s\n',char(verify{n}));    %
    end
    fclose(fid);
    disp('finish aug_verify filepath trans');
    
    % save the aug train in TXT
    fid=fopen(savename4_aug_train, 'w'); % �ļ�cΪ�������Ϻ���ļ�==============
    train = folder_final(iii).aug_train;
    for n=1:length(train)
        fprintf(fid,'%s\n',char(train{n}));    %
    end
    fclose(fid);
    disp('finish aug_train filepath trans'); 
    
    % save the or train in
    % TXT(û������ѵ�������������ںͲ��Լ���֤��һ��predict��)===============
    fid=fopen(savename4_or_train, 'w'); % �ļ�cΪ�������Ϻ���ļ�
    train = folder_final(iii).or_train;
    for n=1:length(train)
        fprintf(fid,'%s\n',char(train{n}));    %
    end
    fclose(fid);
    disp('finish or_train filepath trans'); 
end

save(strcat(savepath,filesep,'mat_folder'),'folder_final','folder','augdict','test_filename_final','aug_filename_final','K_fold'); 














