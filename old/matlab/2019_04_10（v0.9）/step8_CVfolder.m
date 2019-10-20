% ������֤���۴��룬���ΪtxtĿ¼���������ڶ��������⣩
% ��������۲���Ϊ��ÿһ�൥������������֤û����������ƽ��
% ���������������ݴ����Ĵ���
% ע����������ʱ��Ҫһ���ţ���Ҫ�������ȥ��

% �ļ��������򣺵�һλ����Ϊ������ţ��ڶ�λ����Ϊblock��ţ�����λ����Ϊ�������
% ��һ��������������blockĿ¼��augĿ¼��testĿ¼����cell��ע�⣬cut��Ҫ��������룬aug��block��������ֱ��ȫ������һ���ļ��м��ɣ�
% �ڶ������÷������ܵ�ʱ���������ݷŵ�һ���ļ��м��ɣ�ѵ����������aug���ݣ���֤������������Ҳ��aug��һ���ļ��У�test��δ�����ģ�
%%
clear all;
clc;
%% ��������
K_fold = 10;
%rate2verify = 1/(K_fold-1);%��֤���ӷ�test�����ٷֳ���,Ҳ���������������ϣ���֤���Ͳ��Լ����
savepath = 'H:\@data_pnens_zhuanyi_dl\CVfold10';
%% AUG data 4 train and verify
% AUG ��all negative object fold in a cell
AUG_fold_cell{1}='H:\@data_pnens_zhuanyi_dl\aug\a';
% AUG_fold_cell{2}='G:\diploma_project\data_huanhui\@aug_data\G2';
% AUG_fold_cell{3}='G:\diploma_project\data_huanhui\@aug_data\G3';
% p_fold_cell{4}='';
% p_fold_cell{5}='';
% p_fold_cell{6}='';

%% test data
% test ��all negative object fold in a cell
test_fold_cell{1}='H:\@data_pnens_zhuanyi_dl\test\a';
% test_fold_cell{2}='G:\diploma_project\data_huanhui\@test_data\G2';
% test_fold_cell{3}='G:\diploma_project\data_huanhui\@test_data\G3';
% test_fold_cell{4}='';

%% cut data Ϊ�˷��۵õ���ǩ��ǰ�������������۵Ľ��
% cut ��all negative object fold in a cell
n_fold_cell{1}='G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@data\a\1mat\zhuanyi';
% n_fold_cell{2}='';
% n_fold_cell{3}='';
% n_fold_cell{4}='';

% cut ��all positive object fold in a cell
p_fold_cell{1}='G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@data\a\1mat\buzhuanyi';
% p_fold_cell{2}='';
% p_fold_cell{3}='';
% p_fold_cell{4}='';


%% ��ȡ�ļ���
aug_filename = [];
test_filename = [];
cut_n_foldname = [];
cut_p_foldname = [];

aug_filename_final = {};
test_filename_final = {};
cut_n_foldname_final = {};
cut_p_foldname_final = {};

%AUG----------------------------------------------------------------
for ii = 1:size(AUG_fold_cell,2)
    tmp_filename = dir(strcat(AUG_fold_cell{ii},filesep,'*.h5'));
    aug_filename = [aug_filename;tmp_filename];
end
for ii = 1:size(aug_filename,1)
    aug_filename_final{ii}=aug_filename(ii).name;    
end
aug_filename_final = aug_filename_final';

%TEST----------------------------------------------------------------
for ii = 1:size(test_fold_cell,2)
    tmp_filename = dir(strcat(test_fold_cell{ii},filesep,'*.h5'));
    test_filename = [test_filename;tmp_filename];
end
for ii = 1:size(test_filename,1)
    test_filename_final{ii}=test_filename(ii).name;    
end
test_filename_final = test_filename_final';
%nagetive----------------------------------------------------------------
for ii = 1:size(n_fold_cell,2)
    tmp_filename = dir(strcat(n_fold_cell{ii},filesep,'*.mat'));
    cut_n_foldname = [cut_n_foldname;tmp_filename];
end
for ii = 1:size(cut_n_foldname,1)
    cut_n_foldname_final{ii}=cut_n_foldname(ii).name;    
end
cut_n_foldname_final = cut_n_foldname_final';
%positive----------------------------------------------------------------
for ii = 1:size(p_fold_cell,2)
    tmp_filename = dir(strcat(p_fold_cell{ii},filesep,'*.mat'));
    cut_p_foldname = [cut_p_foldname;tmp_filename];
end
for ii = 1:size(cut_p_foldname,1)
    cut_p_foldname_final{ii}=cut_p_foldname(ii).name;    
end
cut_p_foldname_final = cut_p_foldname_final';

%% ����cut�����������ۣ�������Žṹ������
all_subject_name = [cut_n_foldname_final;cut_p_foldname_final];
%ȥ����չ��
for ii = 1:size(cut_n_foldname_final,1)
    tmp = cut_n_foldname_final{ii};
    cut_n_foldname_final{ii}=tmp(1:end-4);
end
for ii = 1:size(cut_p_foldname_final,1)
    tmp = cut_p_foldname_final{ii};
    cut_p_foldname_final{ii}=tmp(1:end-4);
end
for ii = 1:size(all_subject_name,1)
    tmp = all_subject_name{ii};
    all_subject_name{ii}=tmp(1:end-4);
end

%������֤�õ�index
n_index = crossvalind('kfold',size(cut_n_foldname_final,1),K_fold);
p_index = crossvalind('kfold',size(cut_p_foldname_final,1),K_fold);

%����index����������ŷ���fold�ṹ��������test��
for ii = 1:K_fold
    n_tmp_index = (n_index==ii);
    p_tmp_index = (p_index==ii);
    test_data_n = cut_n_foldname_final(n_tmp_index,:);
    test_data_p = cut_p_foldname_final(p_tmp_index,:);
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
    

%% ������Žṹ�����飬����ʵ���ļ�������
for iiii = 1:K_fold
    disp(strcat('running fold:',num2str(iiii)));
    % ��block�ļ�����ѡ��Ӧ�۵�test����
    final_test = {};
    for iii = 1:size(test_filename_final,1)
        tmp_str = test_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(1:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).test)
            final_test = [final_test;tmp_str];
        end 
    end
    folder_final(iiii).test = final_test;
    %��block�ļ�����ѡ��Ӧ�۵�verify����
    final_verify = {};
    for iii = 1:size(test_filename_final,1)
        tmp_str = test_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(1:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).verify)
            final_verify = [final_verify;tmp_str];
        end 
    end
    folder_final(iiii).verify = final_verify;    
    %��aug�ļ�����ѡ��Ӧ�۵�train����
    final_train = {};
    for iii = 1:size(aug_filename_final,1)
        tmp_str = aug_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(1:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).train)
            final_train = [final_train;tmp_str];
        end
    end
    folder_final(iiii).train = final_train;
    
end


%% ���ΪTXT�Թ�pythonʹ��

for iii = 1:K_fold
    savename4test = strcat(savepath,filesep,'fold_',num2str(iii),'_test.txt');
    savename4verify = strcat(savepath,filesep,'fold_',num2str(iii),'_verify.txt');
    savename4train = strcat(savepath,filesep,'fold_',num2str(iii),'_train.txt');
    
    % save the test in TXT
	fid=fopen(savename4test, 'w'); % �ļ�cΪ�������Ϻ���ļ�
    test = folder_final(iii).test;
    for n=1:length(folder_final(iii).test)
        fprintf(fid,'%s\n',char(test{n}));    %  
    end
    fclose(fid);
    disp('finish test filepath trans'); 
    % save the verify in TXT
    fid=fopen(savename4verify, 'w'); % �ļ�cΪ�������Ϻ���ļ�
    verify = folder_final(iii).verify;
    for n=1:length(folder_final(iii).verify)
        fprintf(fid,'%s\n',char(verify{n}));    %
    end
    fclose(fid);
    disp('finish verify filepath trans');
    % save the train in TXT
    fid=fopen(savename4train, 'w'); % �ļ�cΪ�������Ϻ���ļ�
    train = folder_final(iii).train;
    for n=1:length(folder_final(iii).train)
        fprintf(fid,'%s\n',char(train{n}));    %
    end
    fclose(fid);
    disp('finish train filepath trans'); 
end

save(strcat(savepath,filesep,'mat_folder'),'folder_final','folder'); 

















