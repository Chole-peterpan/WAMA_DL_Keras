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
fold_path = 'G:\result\test4';%
fold_name = '2';
a = 1;
indexx = 1;%ʹ��ָ�����ţ�1���ǵ�1ָ��,���ջᰴ�����ָ����ѡ��Ȩ�أ�ָ��Խ���ʾģ��Խ�ã�����lossԽС��ʾģ��Խ�ã�
batch_size = 6;
finf = dir([fold_path,'\*.txt']);
min_test_select_num = 40;%������ѡ�Ĳ��Լ��Ľ���ļ�����С����������ͬiteration��������

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
or_train_loss_path = strcat(fold_path,filesep,'@',fold_name,'_or_loss.txt');
verfiy_loss_path = strcat(fold_path,filesep,'@',fold_name,'_ver_loss.txt');
test_loss_path = strcat(fold_path,filesep,'@',fold_name,'_test_loss.txt');

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
plot(loss_iter,ver_loss,'r');hold on;
plot(loss_iter,test_loss,'g');hold on;
plot(loss_iter,ortrain_loss,'b');hold on;
plot(1:length(smooth(smooth(loss,50))),smooth(smooth(loss,50))/batch_size,'m')

plot(loss_iter,smooth(ver_loss,50),'y');hold on;
plot(loss_iter,smooth(test_loss,50),'r');hold on;
plot(loss_iter,smooth(ortrain_loss,50),'y');
xlabel('iter');
ylabel('loss');
legend({'ver loss','test loss','ortrain loss','minibatch loss','ver loss trend','test loss trend','ortrain loss trend'},'Location','best'); %���ݻ�ͼ���Ⱥ�˳��һ�α�ע���ߡ�
%% step3����ȡverify��������
file_name = strcat(fold_path,filesep,'@',fold_name,'_ver_result.txt');
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
file_name = strcat(fold_path,filesep,'@',fold_name,'_test_result.txt');
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

%% step5�����ݽ�ȡ��    Ϊɶ���ݱ��治ȫ�أ������� fuck
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
ver_result_sort_by_allv = sortrows(ver_result,indexx+1);
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

final_test_result = sortrows(final_test_result,indexx+1);
final_test_result = final_test_result(end:-1:1,:);

final_value = final_test_result(1,:);
test_orlabel_path = strcat(fold_path,filesep,'testorginal_',num2str(final_value(1)),'.txt');
test_prevalue_path = strcat(fold_path,filesep,'testpredict_',num2str(final_value(1)),'.txt');
test_subID_path = strcat(fold_path,filesep,'testsubjectid_',num2str(final_value(1)),'.txt');


test_label = importdata(test_orlabel_path);
test_prevalue = importdata(test_prevalue_path);
test_ID = importdata(test_subID_path);

final_value = final_value(1:6);
a_final_value = final_value;%��a��ͷcopy���£�����鿴��
a_iters = final_value(1);
a_acc = final_value(2);
a_sen = final_value(3);
a_spc = final_value(4);
a_AUC = final_value(5);
a_Loss = final_value(6);
%% save
save(strcat(save_path,filesep,'folder_',fold_name),'test_label','test_prevalue','test_ID','final_value'); 
