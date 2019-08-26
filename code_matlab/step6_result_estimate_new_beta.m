%% ��Ҫ����ָ��
% allv_select_num ����ɸѡ���������µ���֤������
% 

%% step1����ʼ��
clear;
clc;

%% step2��������·������
%��1ָ�꣺acc
%��2ָ�꣺sen
%��3ָ�꣺spc
%��4ָ�꣺auc
%��5ָ�꣺loss
%��6ָ�꣺��acc+sen+spc+auc��/loss   �����ָ����ͺ󣬳���loss
%��7ָ�꣺��acc+sen+spc+auc��+ a/loss ��a��������Լ�����Խ���ʾloss�Ĺ���Խ��
%��8ָ�꣺��acc+sen+spc+auc��
%��9ָ�꣺��acc*sen*spc*auc��/loss 
%��10ָ�꣺��acc*sen*spc*auc��
%��11ָ�꣺��sen+spc��
%��12ָ�꣺��sen+spc��/loss
%��13ָ�꣺��sen*spc��
%��14ָ�꣺��sen*spc��/loss

save_path = 'G:\qweqweqweqwe';
fold_path = 'F:\δ�����ļ��� 2\@test37';%
fold_name = '1';
a = 1;
indexx = 1;%ʹ��ָ�����ţ�1���ǵ�1ָ��,���ջᰴ�����ָ����ѡ��Ȩ�أ�ָ��Խ���ʾģ��Խ�ã�����lossԽС��ʾģ��Խ�ã�
batch_size = 6;
finf = dir([fold_path,'\*.txt']);
min_test_select_num = 40;%������ѡ�Ĳ��Լ��Ľ���ļ�����С����������ͬiteration��������
iiindex = 20000;%���յ���������ֵ����ֹ���ڽ������
exchange_flag = 0;%1��test��ver���ϻ���
%% ��ʾѧϰ������
% lr_path = strcat(fold_path,filesep,'@',fold_name,'_lr.txt');
% lr =  importdata(lr_path);
% figure;
% plot(lr,'b');
% legend({'lr'},'Location','best');
%% ��ʾһ��minibatch_loss

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
legend({'mini loss','smooth level1','smooth level2','smooth level3'},'Location','best');
%% ��ʾ3��loss��������ʾ
if exchange_flag == 0
    tmp_str_t = '_test_loss';
    tmp_str_v = '_ver_loss';
else
    tmp_str_t = '_ver_loss';
    tmp_str_v = '_test_loss';
end



or_train_loss_path = strcat(fold_path,filesep,'@',fold_name,'_or_loss.txt');
verfiy_loss_path = strcat(fold_path,filesep,'@',fold_name,tmp_str_v,'.txt');
test_loss_path = strcat(fold_path,filesep,'@',fold_name,tmp_str_t,'.txt');

or_train_loss_str = importdata(or_train_loss_path);
verfiy_loss_str = importdata(verfiy_loss_path);
test_loss_str = importdata(test_loss_path);


ortrain_loss = [];
ver_loss = [];
test_loss = [];
loss_iter = [];

%���ǵ����������ݱ��治ȫ��������������е�ʱ��ĳʱ���жϣ�����ȡmin��������
for ii= 1:min([length(or_train_loss_str),length(test_loss_str),length(verfiy_loss_str)])
    tmp_str_ortrain = or_train_loss_str{ii};
    tmp_str_ver = verfiy_loss_str{ii};
    tmp_str_test = test_loss_str{ii};

    index = find(tmp_str_ortrain=='@');
    loss_iter(ii) = str2double(tmp_str_ortrain(1:index-1));
    ortrain_loss(ii) = str2double(tmp_str_ortrain(index+1:end));
    
    index = find(tmp_str_ver=='@');
    ver_loss(ii) = str2double(tmp_str_ver(index+1:end));
    
    index = find(tmp_str_test=='@');
    test_loss(ii) = str2double(tmp_str_test(index+1:end));
  
end

figure;
% plot(loss_iter,ver_loss,'r');hold on;
% plot(loss_iter,test_loss,'g');hold on;
% plot(loss_iter,ortrain_loss,'b');hold on;

plot(1:length(smooth(smooth(loss,50))),smooth(smooth(loss,50))/batch_size,'m');hold on;
plot(loss_iter,smooth(ver_loss,50),'r');hold on;
plot(loss_iter,smooth(test_loss,50),'g');hold on;
plot(loss_iter,smooth(ortrain_loss,50),'b');
xlabel('iter');
ylabel('loss');
legend({'minibatch loss','ver loss trend','test loss trend','ortrain loss trend'},'Location','best'); 
% legend({'ver loss','test loss','ortrain loss','minibatch loss','ver loss trend','test loss trend','ortrain loss trend'},'Location','best'); %���ݻ�ͼ���Ⱥ�˳��һ�α�ע���ߡ�

%% step3����ȡverify��������
if exchange_flag == 0
    tmp_str_v = '_ver_result';
else
    tmp_str_v = '_test_result';
end

file_name = strcat(fold_path,filesep,'@',fold_name,tmp_str_v,'.txt');
ver_result_str = importdata(file_name);
ver_result = zeros(length(ver_result_str),6);
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
    index = find(tmp_str==',');
    auc = str2double(tmp_str(1:index(1)-1));
    tmp_str = tmp_str(index(1)+1:end);
    %loss
    index = find(tmp_str==']');
    loss = str2double(tmp_str(1:index(1)-1));
    
    
    ver_result(ii,:) = [iter,acc,sen,spc,auc,loss];  %======================== 
end


%% step4����ȡtest��������
if exchange_flag == 0
    tmp_str_v = '_test_result';
else
    tmp_str_v = '_ver_result';
end

file_name = strcat(fold_path,filesep,'@',fold_name,tmp_str_v,'.txt');
test_result_str = importdata(file_name);
test_result = zeros(length(test_result_str),6);
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
    index = find(tmp_str==',');
    auc = str2double(tmp_str(1:index(1)-1));
    tmp_str = tmp_str(index(1)+1:end);
    %loss
    index = find(tmp_str==']');
    loss = str2double(tmp_str(1:index(1)-1));
    
    test_result(ii,:) = [iter,acc,sen,spc,auc,loss];   
end

%% step5�����ݽ�ȡ����ʱ�����ݻᱣ�治ȫ������ļ����������ᱣ�治������Ϊɶ���ݱ��治ȫ�أ������� fuck
cut_index = min(length(test_result_str),length(ver_result_str));
if length(test_result_str) == cut_index
    ver_result = ver_result(1:cut_index,:);
else
    test_result = test_result(1:cut_index,:);
end
%% step6��������ָ�꣺��������ѡģ�͵�ָ�꣩
%����ָ�꣺======================================
%��1ָ�꣺acc
%��2ָ�꣺sen
%��3ָ�꣺spc
%��4ָ�꣺auc
%��5ָ�꣺loss
%��ָ�꣺���±�7��ʼ����Ϊԭ�����һ����iters===========================================
%��6ָ�꣺��acc+sen+spc+auc��/loss   �����ָ����ͺ󣬳���loss
%��7ָ�꣺��acc+sen+spc+auc��+ a/loss ��a��������Լ�����Խ���ʾloss�Ĺ���Խ��
%��8ָ�꣺��acc+sen+spc+auc��
%��9ָ�꣺��acc*sen*spc*auc��/loss 
%��10ָ�꣺��acc*sen*spc*auc��
%��11ָ�꣺��sen+spc��
%��12ָ�꣺��sen+spc��/loss
%��13ָ�꣺��sen*spc��
%��14ָ�꣺��sen*spc��/loss

%��6ָ��
ver_result(:,7)=(ver_result(:,2)+ver_result(:,3)+ ver_result(:,4)+ver_result(:,5))./ver_result(:,6);
test_result(:,7)=(test_result(:,2)+test_result(:,3)+ test_result(:,4)+test_result(:,5))./test_result(:,6);
%��7ָ��
ver_result(:,8)=(ver_result(:,2)+ver_result(:,3)+ ver_result(:,4)+ver_result(:,5))+a./ver_result(:,6);
test_result(:,8)=(test_result(:,2)+test_result(:,3)+ test_result(:,4)+test_result(:,5))+a./test_result(:,6);
%��8ָ��
ver_result(:,9)=(ver_result(:,2)+ver_result(:,3)+ ver_result(:,4)+ver_result(:,5));
test_result(:,9)=(test_result(:,2)+test_result(:,3)+ test_result(:,4)+test_result(:,5));
%��9ָ��
ver_result(:,10)=(ver_result(:,2).*ver_result(:,3).* ver_result(:,4).*ver_result(:,5))./ver_result(:,6);
test_result(:,10)=(test_result(:,2).*test_result(:,3).* test_result(:,4).*test_result(:,5))./test_result(:,6);
%��10ָ��
ver_result(:,11)=(ver_result(:,2).*ver_result(:,3).* ver_result(:,4).*ver_result(:,5));
test_result(:,11)=(test_result(:,2).*test_result(:,3).* test_result(:,4).*test_result(:,5));  
%��11ָ��
ver_result(:,12)=(ver_result(:,3)+ ver_result(:,4));
test_result(:,12)=(test_result(:,3)+ test_result(:,4));  
%��12ָ��
ver_result(:,13)=(ver_result(:,3)+ ver_result(:,4))./ver_result(:,6);
test_result(:,13)=(test_result(:,3)+ test_result(:,4))./test_result(:,6); 
%��13ָ��
ver_result(:,14)=(ver_result(:,3).* ver_result(:,4));
test_result(:,14)=(test_result(:,3).* test_result(:,4));  
%��14ָ��
ver_result(:,15)=(ver_result(:,3).* ver_result(:,4))./ver_result(:,6);
test_result(:,15)=(test_result(:,3).* test_result(:,4))./test_result(:,6); 


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
%% step7�����յ�indexxָ��ɸѡ����iter
%���յ�indexxָ�����򲢽�ȡ��ȡ��Ϊallv
%==========================================������������������������������

ver_result = ver_result(ver_result(:,1)>iiindex,:);%���Ȱ�С��һ���������Ŀ���


ver_result_sort_by_allv = sortrows(ver_result,indexx+1);
ver_result_sort_by_allv = ver_result_sort_by_allv(end:-1:1,:);

                        
allv_value = [];
for ii = 1:size(ver_result_sort_by_allv,1)
   if ~(ismember(ver_result_sort_by_allv(ii,indexx+1),allv_value))
       allv_value = [allv_value,ver_result_sort_by_allv(ii,indexx+1)];
   end
end
allv_value = sort(allv_value,'descend');

allv_select_num = 0;%����accɸѡ����֤������ļ�������
for ii = 1:length(allv_value)
   tmp_index =  find(ver_result_sort_by_allv(:,indexx+1)==allv_value(ii));
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

final_test_result = sortrows(final_test_result,indexx+1);
final_test_result = final_test_result(end:-1:1,:);


%% ������test��������Ӧ��Ԥ��ֵȫ����������labelҲ������
if exchange_flag == 0
    tmp_str_s = 'test';
else
    tmp_str_s = 'vre';
end


b_test_label_new = [];
b_test_pre_value_new = [];
b_pp_id = [];
for i = 1:size(final_test_result)
    orlabel_path = strcat(fold_path,filesep,tmp_str_s,'orginal_',num2str(final_test_result(i,1)),'.txt');
    preval_path = strcat(fold_path,filesep,tmp_str_s,'predict_',num2str(final_test_result(i,1)),'.txt');
    ID_path = strcat(fold_path,filesep,tmp_str_s,'subjectid_',num2str(final_test_result(i,1)),'.txt');
    b_test_label_new = [b_test_label_new;(importdata(orlabel_path))'];
    b_test_pre_value_new = [b_test_pre_value_new;(importdata(preval_path))'];
    b_pp_id = [b_pp_id;(importdata(ID_path))'];
end


%% ����ȡ��ֵ���൱�ڰѺܶ����������������,��������������ʵ����ǰ���min_test_select_num��
test_ID = b_pp_id(1,:);
test_prevalue = mean(b_test_pre_value_new);
test_label = b_test_label_new(1,:);

[X,Y,T,AUC] = perfcurve(test_label,test_prevalue,1);
[ A_sensitivity, A_specificity] = sen2spc( test_label' ,double(test_prevalue>=0.5)');
accc = sum(test_label == double(test_prevalue>=0.5))/length(test_label);
%% d


a_iters = mean(final_test_result(:,1));
a_acc = accc;
a_sen = A_sensitivity;
a_spc = A_specificity;
a_AUC = AUC;
a_Loss = mean(final_test_result(:,6));

final_value = [a_iters,a_acc,a_sen,a_spc,a_AUC];
%% save
save(strcat(save_path,filesep,'folder_',fold_name),'test_label','test_prevalue','test_ID','final_value'); 