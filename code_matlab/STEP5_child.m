
%% 合成folder
% 读入label与id
wkspace = load([block_data,filesep,'subject',filesep,'subject']);
id = wkspace.xls_data(:,1);
label = wkspace.xls_data(:,label_index+1);

% 获取该label中各个类，以及各类包含样本的id，和数量
all_class = unique(label); %所有类别的名称
per_class_id = {}; % 各类别包含的id
per_class_num = []; % 各类别包含的病人数量
for i = 1:length(all_class)
    per_class_id{end+1} = id(label == all_class(i));
    per_class_num(end+1) = length(per_class_id{end});
end

% 检查：如果某一类的数量小于分折数，则error（因为最多就是留一法）
for i = 1:length(per_class_num)
   if per_class_num(i) < K
       error(['class ',num2str(all_class(i)), ' do not have enough subjects for ',num2str(K),' fold']);
   end
end


%% 获得根据病人分折的结构体数组 folder
%交叉验证得到index
per_class_index = {};
for i = 1:length(all_class)
    per_class_index{end+1}= crossvalind('kfold',per_class_num(i),K);
end

% 构建folder
folder = [];
for i = 1:K
    test = [];
    for ii = 1:length(all_class)
       test = [test;per_class_id{ii}(per_class_index{ii}==i)];
    end
    folder(i).test = test;   
end
folder(1).verify = folder(K).test;
for i = 2:K
    folder(i).verify = folder(i-1).test; 
end
for i = 1:K
    train = [];
    for ii = 1:length(id)
       if  ~ismember(id(ii), folder(i).verify)  &&  ~ismember(id(ii), folder(i).test)
          train = [train;id(ii)];
       end
    end
    folder(i).train = train; 
end
% 将folder转换为cell储存
folder_cell = [];
for i = 1:K
    test = {};
    for ii = 1:length(folder(i).test)
       test{end+1}= num2str(folder(i).test(ii));
    end
    folder_cell(i).test = test';
    
    verify = {};
    for ii = 1:length(folder(i).verify)
       verify{end+1}= num2str(folder(i).verify(ii));
    end
    folder_cell(i).verify = verify';
    
        train = {};
    for ii = 1:length(folder(i).train)
       train{end+1}= num2str(folder(i).train(ii));
    end
    folder_cell(i).train = train';
end
folder = folder_cell;

%% 如果已经有了folder，那么就读取并覆盖
if have_folder
    folder_space = load(folder_file);
    folder = folder_space.folder;
end


%% 根据序号结构体数组，构建实际文件名数组
test_filename_final  ={};
or_filename = dir(strcat(or_data,filesep,'*.h5'));
for ii = 1:length(or_filename)
    test_filename_final{ii}=or_filename(ii).name;    
end
test_filename_final = test_filename_final';


aug_filename_final = {};
aug_filename = dir(strcat(aug_data,filesep,'*.h5'));
for ii = 1:length(aug_filename)
    aug_filename_final{ii}=aug_filename(ii).name;    
end
aug_filename_final = aug_filename_final';


folder_final = [];
for iiii = 1:K
    disp(strcat('making fold:',num2str(iiii),'...'));
    % ==========================================================
    % test样本对应的h5文件名
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
    % ver样本对应的h5文件名
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
    % train样本对应的h5文件名
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


%% 输出为TXT以供python使用

for iii = 1:K
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

save(strcat(savepath,filesep,'mat_folder'),'folder_final','folder','test_filename_final','aug_filename_final','K'); 


