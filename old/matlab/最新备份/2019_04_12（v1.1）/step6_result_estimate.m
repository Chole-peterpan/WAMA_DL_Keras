%% step1����ʼ��
clear;
clc;
save_path = 'H:\dip\diploma\result\mat\test';


%% step2��������·������
fold_path = 'H:\dip\diploma\result\diploma\@pnens_zhuanyi_resnet_a\pnens_zhuanyi_fold5_20190218';%
fold_name = '5';
finf = dir([fold_path,'\*.txt']);
min_test_select_num = 120;%������ѡ�Ĳ��Լ��Ľ���ļ�����С����������ͬiteration��������

%% ��ʾһ��loss

loss_path = strcat(fold_path,filesep,'@',fold_name,'_loss.txt');
loss =  importdata(loss_path);
figure;
plot(loss,'b');
hold on;
plot(smooth(loss),'r');
hold on;
plot(smooth(smooth(loss,8)),'g');
hold on;
plot(smooth(smooth(loss,50)),'m');




%% step3����ȡverify��������
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
    
    ver_result(ii,:) = [iter,acc,sen,spc,auc];  %======================== 
end


%% step4����ȡtest��������
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

%% step5�����ݽ�ȡ��    Ϊɶ���ݱ��治ȫ�أ������� fuck
cut_index = min(length(test_result_str),length(ver_result_str));
if length(test_result_str) == cut_index
    ver_result = ver_result(1:cut_index,:);
else
    test_result = test_result(1:cut_index,:);
end
%% step6����5ָ�깹����������ָ�����
ver_result(:,6)=ver_result(:,2)+ver_result(:,3)+...
                ver_result(:,4)+ver_result(:,5);
test_result(:,6)=test_result(:,2)+test_result(:,3)+...
                 test_result(:,4)+test_result(:,5);
                        
%% xianshiyixia 
figure;
subplot(5,1,1);
plot(ver_result(:,1),ver_result(:,2),'b');hold on;
plot(ver_result(:,1),smooth(ver_result(:,2),20),'r');
title('ver acc');
subplot(5,1,2);
plot(ver_result(:,1),ver_result(:,3),'b');hold on;
plot(ver_result(:,1),smooth(ver_result(:,3),20),'r');
title('sen');
subplot(5,1,3);
plot(ver_result(:,1),ver_result(:,4),'b');hold on;
plot(ver_result(:,1),smooth(ver_result(:,4),20),'r');
title('spc');
subplot(5,1,4);
plot(ver_result(:,1),ver_result(:,5),'b');hold on;
plot(ver_result(:,1),smooth(ver_result(:,5),20),'r');
title('auc');
subplot(5,1,5);
plot(ver_result(:,1),ver_result(:,6),'b');hold on;
plot(ver_result(:,1),smooth(ver_result(:,6),20),'r');
title('sum');
figure;
plot(ver_result(:,1),ver_result(:,2).*ver_result(:,3).*ver_result(:,4).*ver_result(:,5),'b');hold on;
plot(ver_result(:,1),smooth(ver_result(:,2).*ver_result(:,3).*ver_result(:,4).*ver_result(:,5),20),'r');
title('ver miul');

figure;
subplot(5,1,1);
plot(test_result(:,1),test_result(:,2),'b');hold on;
plot(test_result(:,1),smooth(test_result(:,2),20),'r');
title('test acc');
subplot(5,1,2);
plot(test_result(:,1),test_result(:,3),'b');hold on;
plot(test_result(:,1),smooth(test_result(:,3),20),'r');
title('sen');
subplot(5,1,3);
plot(test_result(:,1),test_result(:,4),'b');hold on;
plot(test_result(:,1),smooth(test_result(:,4),20),'r');
title('spc');
subplot(5,1,4);
plot(test_result(:,1),test_result(:,5),'b');hold on;
plot(test_result(:,1),smooth(test_result(:,5),20),'r');
title('auc');
subplot(5,1,5);
plot(test_result(:,1),test_result(:,6),'b');hold on;
plot(test_result(:,1),smooth(test_result(:,6),20),'r');
title('sum');
figure;
plot(test_result(:,1),test_result(:,2).*test_result(:,3).*test_result(:,4).*test_result(:,5),'b');hold on;
plot(test_result(:,1),smooth(test_result(:,2).*test_result(:,3).*test_result(:,4).*test_result(:,5),20),'r');
title('test miul');
%% step7�����ղ�ָͬ��ɸѡ����iter

% %����acc���򲢽�ȡ����ʱ����Ҫ�õ������õ�5ָ�꣩=============================
% ver_result_sort_by_acc = sortrows(ver_result,2);
% ver_result_sort_by_acc = ver_result_sort_by_acc(end:-1:1,:);
% 
%                         
% acc_value = [];
% for ii = 1:size(ver_result_sort_by_acc,1)
%    if ~(ismember(ver_result_sort_by_acc(ii,2),acc_value))
%        acc_value = [acc_value,ver_result_sort_by_acc(ii,2)];
%    end
% end
% acc_value = sort(acc_value,'descend');
% 
% acc_select_num = 0;%����accɸѡ����֤������ļ�������
% for ii = 1:length(acc_value)
%    tmp_index =  find(ver_result_sort_by_acc(:,2)==acc_value(ii));
%    acc_select_num = acc_select_num+length(tmp_index);
%    if acc_select_num >= min_test_select_num
%       break; 
%    end
% end
% 
% %��ȡʱ��������С��������Ϊһ��ָ��ͬһ�������ܶ�Ӧ�ܶ��ļ�������ȡ�಻ȥ��
% result_by_acc = ver_result_sort_by_acc(1:acc_select_num,:);
% %����acc���򲢽�ȡ==========================================================

%���յ�5ָ�����򲢽�ȡ��ȡ��Ϊallv ==========================================
ver_result_sort_by_allv = sortrows(ver_result,6);
ver_result_sort_by_allv = ver_result_sort_by_allv(end:-1:1,:);

                        
allv_value = [];
for ii = 1:size(ver_result_sort_by_allv,1)
   if ~(ismember(ver_result_sort_by_allv(ii,6),allv_value))
       allv_value = [allv_value,ver_result_sort_by_allv(ii,6)];
   end
end
allv_value = sort(allv_value,'descend');

allv_select_num = 0;%����accɸѡ����֤������ļ�������
for ii = 1:length(allv_value)
   tmp_index =  find(ver_result_sort_by_allv(:,6)==allv_value(ii));
   allv_select_num = allv_select_num+length(tmp_index);
   if allv_select_num >= min_test_select_num
      break; 
   end
end

result_by_allv = ver_result_sort_by_allv(1:allv_select_num,:);
%��Ϊһ��ָ��ͬһ�������ܶ�Ӧ�ܶ��ļ�������ȡ�಻ȥ��(������һ�����ָ���ѧ����)����Ϊadapt ģʽ
%result_by_allv = ver_result_sort_by_allv(1:min_test_select_num,:);%ֱ��ȡ��Сֵ
%���յ�5ָ�����򲢽�ȡ��=====================================================


%% step8������ɸѡ���iter����ȡtest���ϵ�����ָ��
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
test_subID_path = strcat(fold_path,filesep,'testsubjectid_',num2str(final_value(1)),'.txt');


test_label = importdata(test_orlabel_path);
test_prevalue = importdata(test_prevalue_path);
test_ID = importdata(test_subID_path);

%% save
save(strcat(save_path,filesep,'folder_',fold_name),'test_label','test_prevalue','test_ID','final_value'); 
