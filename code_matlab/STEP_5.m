%% 交叉验证分折代码，输出为txt目录（仅适用于二分类问题）
% 本代码分折策略为：每一类单独分折数，保证没折样本基本平衡
% 仅适用于老王数据处理后的代码
% 注意给样本编号时需要一起编号，不要分类各自去编

% 文件命名规则：第一位数字为样本序号，第二位数字为block序号，第三位数字为扩增序号
% 第一步：将所有样本block目录和aug目录和test目录输入cell（注意，cut需要分类别输入，aug和block无需分类别，直接全部放在一个文件夹即可）
% 第二步：用服务器跑的时候，所有数据放到一个文件夹即可（训练集即所有aug数据，验证用扩增的所以也和aug是一个文件夹，test用未扩增的）
%%
clear
clc
%% 参数设置
K_fold = 10;
%rate2verify = 1/(K_fold-1);%验证集从非test集里再分出来,也就是在样本数量上，验证机和测试集相等
savepath = 'H:\@data_NENs_recurrence\PNENs\data\@flow1\5folder';
%% AUG data（h5文件）
% AUG ：all object fold in a cell
AUG_fold_cell{1}='H:\@data_NENs_recurrence\PNENs\data\@flow1\4aug_h5';
% AUG_fold_cell{2}='G:\diploma_project\data_huanhui\@aug_data\G2';
% AUG_fold_cell{3}='G:\diploma_project\data_huanhui\@aug_data\G3';
% p_fold_cell{4}='';
% p_fold_cell{5}='';
% p_fold_cell{6}='';

%% or data（h5文件）
% test ：all object fold in a cell
test_fold_cell{1}='H:\@data_NENs_recurrence\PNENs\data\@flow1\3or_h5';
% test_fold_cell{2}='G:\diploma_project\data_huanhui\@test_data\G2';
% test_fold_cell{3}='G:\diploma_project\data_huanhui\@test_data\G3';
% test_fold_cell{4}='';

%% cut data 为了分折得到标签，前两个都是最后分折的结果(mat文件)
% cut ：all negative object fold in a cell
% n_fold_cell{1}='H:\@data_NENs_recurrence\PNENs\data\@flow1\1mat\a';
% n_fold_cell{2}='';
% n_fold_cell{3}='';
% n_fold_cell{4}='';

% cut ：all positive object fold in a cell
% p_fold_cell{1}='H:\@data_NENs_recurrence\PNENs\data\@flow1\1mat\b';
% p_fold_cell{2}='';
% p_fold_cell{3}='';
% p_fold_cell{4}='';


augdict.class_a_id = 1:40;% 手动传入a类病人的id
augdict.class_b_id = 50:59;% 手动传入b类病人的id

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


%% 提取文件名
aug_h5_filename = [];
or_h5_filename = [];
% cut_n_foldname = [];
% cut_p_foldname = [];

aug_filename_final = {};
test_filename_final = {};
% cut_n_foldname_final = {};
% cut_p_foldname_final = {};

%AUG----------------------------------------------------------------
for ii = 1:size(AUG_fold_cell,2)
    tmp_filename = dir(strcat(AUG_fold_cell{ii},filesep,'*.h5'));
    aug_h5_filename = [aug_h5_filename;tmp_filename];
end
for ii = 1:size(aug_h5_filename,1)
    aug_filename_final{ii}=aug_h5_filename(ii).name;    
end
aug_filename_final = aug_filename_final';

%TEST----------------------------------------------------------------
for ii = 1:size(test_fold_cell,2)
    tmp_filename = dir(strcat(test_fold_cell{ii},filesep,'*.h5'));
    or_h5_filename = [or_h5_filename;tmp_filename];
end
for ii = 1:size(or_h5_filename,1)
    test_filename_final{ii}=or_h5_filename(ii).name;    
end
test_filename_final = test_filename_final';
%nagetive----------------------------------------------------------------
% for ii = 1:size(n_fold_cell,2)
%     tmp_filename = dir(strcat(n_fold_cell{ii},filesep,'*.mat'));
%     cut_n_foldname = [cut_n_foldname;tmp_filename];
% end
% for ii = 1:size(cut_n_foldname,1)
%     cut_n_foldname_final{ii}=cut_n_foldname(ii).name;    
% end
% cut_n_foldname_final = cut_n_foldname_final';
%positive----------------------------------------------------------------
% for ii = 1:size(p_fold_cell,2)
%     tmp_filename = dir(strcat(p_fold_cell{ii},filesep,'*.mat'));
%     cut_p_foldname = [cut_p_foldname;tmp_filename];
% end
% for ii = 1:size(cut_p_foldname,1)
%     cut_p_foldname_final{ii}=cut_p_foldname(ii).name;    
% end
% cut_p_foldname_final = cut_p_foldname_final';

%% 根据cut的数据名分折，构建序号结构体数组
% all_subject_name = [cut_n_foldname_final;cut_p_foldname_final];
% %去除扩展名
% for ii = 1:size(cut_n_foldname_final,1)
%     tmp = cut_n_foldname_final{ii};
%     cut_n_foldname_final{ii}=tmp(1:end-4);
% end
% for ii = 1:size(cut_p_foldname_final,1)
%     tmp = cut_p_foldname_final{ii};
%     cut_p_foldname_final{ii}=tmp(1:end-4);
% end
% 
% for ii = 1:size(all_subject_name,1)
%     tmp = all_subject_name{ii};
%     all_subject_name{ii}=tmp(1:end-4);
% end

%交叉验证得到index
n_index = crossvalind('kfold',size(cut_n_name_final,1),K_fold);
p_index = crossvalind('kfold',size(cut_p_name_final,1),K_fold);

%根据index，将样本序号放入fold结构体数组中test中
for ii = 1:K_fold
    n_tmp_index = (n_index==ii);
    p_tmp_index = (p_index==ii);
    test_data_n = cut_n_name_final(n_tmp_index,:);
    test_data_p = cut_p_name_final(p_tmp_index,:);
    folder(ii).test = [test_data_n;test_data_p];
end

%构建结构体中verify部分
for ii = 1:K_fold-1
    folder(ii).verify = folder(ii+1).test;
end
folder(K_fold).verify = folder(1).test;

%构建结构体中train部分
for ii = 1:K_fold
    tmp_file = {};
    for iii = 1:size(all_subject_name,1)
       if  ~ismember(all_subject_name{iii},folder(ii).test) && ~ismember(all_subject_name{iii},folder(ii).verify) 
          tmp_file = [tmp_file;all_subject_name{iii}];
       end
    end 
    folder(ii).train = tmp_file; 
end
    

%% 根据序号结构体数组，构建实际文件名数组
folder_final = [];
for iiii = 1:K_fold
    disp(strcat('running fold:',num2str(iiii)));
    % ==========================================================
    % test样本对应的h5文件名
    or_test = {};
    for iii = 1:size(test_filename_final,1)
        tmp_str = test_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(1:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).test)
            or_test = [or_test;tmp_str];
        end 
    end
    folder_final(iiii).or_test = or_test;
    
    aug_test = {};
    for iii = 1:size(aug_filename_final,1)
        tmp_str = aug_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(1:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).test)
            aug_test = [aug_test;tmp_str];
        end
    end
    folder_final(iiii).aug_test = aug_test;
    % ==========================================================
    % ver样本对应的h5文件名
    or_verify = {};
    for iii = 1:size(test_filename_final,1)
        tmp_str = test_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(1:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).verify)
            or_verify = [or_verify;tmp_str];
        end 
    end
    folder_final(iiii).or_verify = or_verify; 
    
    aug_verify = {};
    for iii = 1:size(aug_filename_final,1)
        tmp_str = aug_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(1:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).verify)
            aug_verify = [aug_verify;tmp_str];
        end
    end
    folder_final(iiii).aug_verify = aug_verify;
    % ==========================================================
    % train样本对应的h5文件名
    or_train = {};
    for iii = 1:size(test_filename_final,1)
        tmp_str = test_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(1:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).train)
            or_train = [or_train;tmp_str];
        end
    end
    folder_final(iiii).or_train = or_train;
    
    aug_train = {};
    for iii = 1:size(aug_filename_final,1)
        tmp_str = aug_filename_final{iii};
        tmp_index = find(tmp_str == '_');
        tmp_str_p = tmp_str(1:tmp_index(1)-1);
        if  ismember(tmp_str_p,folder(iiii).train)
            aug_train = [aug_train;tmp_str];
        end
    end
    folder_final(iiii).aug_train = aug_train;
    % ==========================================================
    % 从block文件中挑选对应折的train数据（未扩增的文件，用来测试网路在整个未扩增训练集上的表现，在测试测试集验证集的时候一起测）
    % 因为代码结构就是训练一定次数，测试三个集合，然后可以观察loss之类的。
%     or_train = {};
%     for iii = 1:size(test_filename_final,1)
%         tmp_str = test_filename_final{iii};
%         tmp_index = find(tmp_str == '_');
%         tmp_str_p = tmp_str(1:tmp_index(1)-1);
%         if  ismember(tmp_str_p,folder(iiii).train)
%             or_train = [or_train;tmp_str];
%         end
%     end
%     folder_final(iiii).or_train = or_train;
    
end


%% 输出为TXT以供python使用

for iii = 1:K_fold
    savename4_or_test = strcat(savepath,filesep,'fold_',num2str(iii),'_or_test.txt');
    savename4_aug_test = strcat(savepath,filesep,'fold_',num2str(iii),'_aug_test.txt');
    savename4_or_verify = strcat(savepath,filesep,'fold_',num2str(iii),'_or_verify.txt');
    savename4_aug_verify = strcat(savepath,filesep,'fold_',num2str(iii),'_aug_verify.txt');
    savename4_or_train = strcat(savepath,filesep,'fold_',num2str(iii),'_or_train.txt');
    savename4_aug_train = strcat(savepath,filesep,'fold_',num2str(iii),'_aug_train.txt');
  
    

    % save the test in TXT
	fid=fopen(savename4_or_test, 'w'); % 文件c为数据整合后的文件===============
    test = folder_final(iii).or_test;
    for n=1:length(test)
        fprintf(fid,'%s\n',char(test{n}));    %  
    end
    fclose(fid);
    disp('finish or_test filepath trans'); 
    
    fid=fopen(savename4_aug_test, 'w'); % 文件c为数据整合后的文件===============
    test = folder_final(iii).aug_test;
    for n=1:length(test)
        fprintf(fid,'%s\n',char(test{n}));    %
    end
    fclose(fid);
    disp('finish aug_test filepath trans');
    
    
    
    % save the verify in TXT
    fid=fopen(savename4_or_verify, 'w'); % 文件c为数据整合后的文件=============
    verify = folder_final(iii).or_verify;
    for n=1:length(verify)
        fprintf(fid,'%s\n',char(verify{n}));    %
    end
    fclose(fid);
    disp('finish or_verify filepath trans');
    
    fid=fopen(savename4_aug_verify, 'w'); % 文件c为数据整合后的文件=============
    verify = folder_final(iii).aug_verify;
    for n=1:length(verify)
        fprintf(fid,'%s\n',char(verify{n}));    %
    end
    fclose(fid);
    disp('finish aug_verify filepath trans');
    
    % save the aug train in TXT
    fid=fopen(savename4_aug_train, 'w'); % 文件c为数据整合后的文件==============
    train = folder_final(iii).aug_train;
    for n=1:length(train)
        fprintf(fid,'%s\n',char(train{n}));    %
    end
    fclose(fid);
    disp('finish aug_train filepath trans'); 
    
    % save the or train in
    % TXT(没扩增的训练集，单纯用于和测试集验证集一起predict的)===============
    fid=fopen(savename4_or_train, 'w'); % 文件c为数据整合后的文件
    train = folder_final(iii).or_train;
    for n=1:length(train)
        fprintf(fid,'%s\n',char(train{n}));    %
    end
    fclose(fid);
    disp('finish or_train filepath trans'); 
end

save(strcat(savepath,filesep,'mat_folder'),'folder_final','folder','augdict','test_filename_final','aug_filename_final','K_fold'); 

















