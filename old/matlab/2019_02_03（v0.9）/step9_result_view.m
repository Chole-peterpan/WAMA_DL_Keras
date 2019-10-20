%% step1：初始化
clear;
clc;
save_path = 'G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@result\pnens_zhuanyi_0219\all\pnens_zhuanyi\mat_result';
min_test_select_num = 120;%最终挑选的测试集的结果文件的最小数量（即不同iteration的数量）

%% step2：参数和路径设置
fold_path = 'G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@result\pnens_zhuanyi_0219\all\pnens_zhuanyi\pnens_zhuanyi_fold5_20190218';
fold_name = '5';
finf = dir([fold_path,'\*.txt']);
%% step3：读取verify的总数据
file_name = strcat(fold_path,filesep,'@',fold_name,'_ver_result.txt');
ver_result_str = importdata(file_name);
ver_result = zeros(length(ver_result_str),5);
for ii = 1:length(ver_result_str)
    tmp_str = ver_result_str{ii};
    %iter
    index = find(tmp_str=='@');
    iter = str2double(tmp_str(1:index-1));
    tmp_str = tmp_str(index+2:end);
    %acc
    index = find(tmp_str==',');
    acc = str2double(tmp_str(1:index(1)-1));
    tmp_str = tmp_str(index(1)+1:end);  
    %sen
    index = find(tmp_str==',');
    sen = str2double(tmp_str(1:index(1)-1));
    tmp_str = tmp_str(index(1)+1:end);
    %spc
    index = find(tmp_str==',');
    spc = str2double(tmp_str(1:index(1)-1));
    tmp_str = tmp_str(index(1)+1:end);
    %auc
    index = find(tmp_str==']');
    auc = str2double(tmp_str(1:index(1)-1));
    
    ver_result(ii,:) = [iter,acc,sen,spc,auc];   
end


%% step4：读取test的总数据
file_name = strcat(fold_path,filesep,'@',fold_name,'_test_result.txt');
test_result_str = importdata(file_name);
test_result = zeros(length(test_result_str),5);
for ii = 1:length(test_result_str)
    tmp_str = test_result_str{ii};
    %iter
    index = find(tmp_str=='@');
    iter = str2double(tmp_str(1:index-1));
    tmp_str = tmp_str(index+2:end);
    %acc
    index = find(tmp_str==',');
    acc = str2double(tmp_str(1:index(1)-1));
    tmp_str = tmp_str(index(1)+1:end);  
    %sen
    index = find(tmp_str==',');
    sen = str2double(tmp_str(1:index(1)-1));
    tmp_str = tmp_str(index(1)+1:end);
    %spc
    index = find(tmp_str==',');
    spc = str2double(tmp_str(1:index(1)-1));
    tmp_str = tmp_str(index(1)+1:end);
    %auc
    index = find(tmp_str==']');
    auc = str2double(tmp_str(1:index(1)-1));
    
    test_result(ii,:) = [iter,acc,sen,spc,auc];   
end

%% step5：数据截取：    为啥数据保存不全呢？？？？ fuck
cut_index = min(length(test_result_str),length(ver_result_str));
if length(test_result_str) == cut_index
    ver_result = ver_result(1:cut_index,:);
else
    test_result = test_result(1:cut_index,:);
end
%% step6：第5指标构建：即所有指标求和
ver_result(:,6)=ver_result(:,2)+ver_result(:,3)+...
                ver_result(:,4)+ver_result(:,5);
test_result(:,6)=test_result(:,2)+test_result(:,3)+...
                 test_result(:,4)+test_result(:,5);
                        

%% step7：按照不同指标筛选最终iter

%按照acc排序并截取（暂时不需要用到，先用第5指标）=============================
ver_result_sort_by_acc = sortrows(ver_result,2);
ver_result_sort_by_acc = ver_result_sort_by_acc(end:-1:1,:);

                        
acc_value = [];
for ii = 1:size(ver_result_sort_by_acc,1)
   if ~(ismember(ver_result_sort_by_acc(ii,2),acc_value))
       acc_value = [acc_value,ver_result_sort_by_acc(ii,2)];
   end
end
acc_value = sort(acc_value,'descend');

acc_select_num = 0;%经过acc筛选的验证集结果文件的数量
for ii = 1:length(acc_value)
   tmp_index =  find(ver_result_sort_by_acc(:,2)==acc_value(ii));
   acc_select_num = acc_select_num+length(tmp_index);
   if acc_select_num >= min_test_select_num
      break; 
   end
end

%截取时不按照最小数量，因为一个指标同一个数可能对应很多文件，尽量取多不去少
result_by_acc = ver_result_sort_by_acc(1:acc_select_num,:);
%按照acc排序并截取==========================================================

%按照第5指标排序并截取：取名为allv ==========================================
ver_result_sort_by_allv = sortrows(ver_result,6);
ver_result_sort_by_allv = ver_result_sort_by_allv(end:-1:1,:);

                        
allv_value = [];
for ii = 1:size(ver_result_sort_by_allv,1)
   if ~(ismember(ver_result_sort_by_allv(ii,6),allv_value))
       allv_value = [allv_value,ver_result_sort_by_allv(ii,6)];
   end
end
allv_value = sort(allv_value,'descend');

allv_select_num = 0;%经过acc筛选的验证集结果文件的数量
for ii = 1:length(allv_value)
   tmp_index =  find(ver_result_sort_by_allv(:,6)==allv_value(ii));
   allv_select_num = allv_select_num+length(tmp_index);
   if allv_select_num >= min_test_select_num
      break; 
   end
end

result_by_allv = ver_result_sort_by_allv(1:allv_select_num,:);
%按照第5指标排序并截取：=====================================================


%% step8：按照筛选后的iter，提取test集合的所有指标
ilter_final = result_by_allv(:,1);

final_test_result = [];
for ii = 1:length(ilter_final)
   index = find(test_result(:,1)==ilter_final(ii)); 
    final_test_result = [final_test_result;test_result(index,:)];
end

final_test_result = sortrows(final_test_result,6);
final_test_result = final_test_result(end:-1:1,:);

final_value = final_test_result(1,:);
test_orlabel_path = strcat(fold_path,filesep,'testorginal_',num2str(final_value(1)),'.txt');
test_prevalue_path = strcat(fold_path,filesep,'testpredict_',num2str(final_value(1)),'.txt');


test_label = importdata(test_orlabel_path);
test_prevalue = importdata(test_prevalue_path);


%% save
save(strcat(save_path,filesep,'folder_',fold_name),'test_label','test_prevalue','final_value'); 


































