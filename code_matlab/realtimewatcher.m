%% ��Ҫ����ָ��
% allv_select_num ����ɸѡ���������µ���֤������
%
% while(1)
%% step1����ʼ��
clear;
clc;
close all
%% step2�������·������
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
%��15ָ�꣺sen+auc


save_path = '/data/XS_Aug_model_result/model_templete/recurrent/test_waibu';
log_path = '/data/XS_Aug_model_result/model_templete/recurrent/test_waibu/test_waibu9/log';%
fold_name = '1';
a = 2;
indexx = 8;%ʹ��ָ�����ţ�1���ǵ�1ָ��,���ջᰴ�����ָ����ѡ��Ȩ�أ�ָ��Խ���ʾģ��Խ�ã�����lossԽС��ʾģ��Խ�ã�
batch_size = 1;
% finf = dir([fold_path,'\*.txt']);
min_test_select_num = 40;%������ѡ�Ĳ��Լ��Ľ���ļ�����С����������ͬiteration��������
threshold_iter = 39999;%��ǰn�ε���ģ��ʦ���ɿ��ģ�������Ҫ�ֶ����õ������ֵ��ǰn�εĽ���
exchange_flag = 0;%1��test��ver���ϻ���
%% ��ʾѧϰ������
lr_path = strcat(log_path,filesep,'@',fold_name,'_lr.txt');
lr =  importdata(lr_path);
figure;
plot(lr,'b');
legend({'lr'},'Location','best');
%% ��ʾһ��minibatch_loss

minibatch_loss_path = strcat(log_path,filesep,'@',fold_name,'_loss.txt');
minibatch_loss =  importdata(minibatch_loss_path);
figure;
plot(minibatch_loss,'b');
hold on;
plot(smooth(minibatch_loss),'r');
hold on;
plot(smooth(smooth(minibatch_loss,8)),'g');
hold on;
plot(smooth(smooth(minibatch_loss,50)),'m');
legend({'mini loss','smooth level1','smooth level2','smooth level3'},'Location','best');
%% ��ʾ4��loss��������ʾ
if exchange_flag == 0
    tmp_str_t = '_test_loss';
    tmp_str_v = '_ver_loss';
else
    tmp_str_t = '_ver_loss';
    tmp_str_v = '_test_loss';
end



ortrain_loss_path = strcat(log_path,filesep,'@',fold_name,'_or_loss.txt');
verfiy_loss_path = strcat(log_path,filesep,'@',fold_name,tmp_str_v,'.txt');
test_loss_path = strcat(log_path,filesep,'@',fold_name,tmp_str_t,'.txt');

or_train_loss_str = importdata(ortrain_loss_path);
verfiy_loss_str = importdata(verfiy_loss_path);
test_loss_str = importdata(test_loss_path);


ortrain_loss = [];
ver_loss = [];
test_loss = [];
loss_iter = [];

%���ǵ���������ݱ��治ȫ��������������е�ʱ��ĳʱ���жϣ�����ȡmin�������
for ii= 1:max([length(or_train_loss_str),length(test_loss_str),length(verfiy_loss_str)])
%     tmp_str_ortrain = or_train_loss_str{ii};
%     tmp_str_ver = verfiy_loss_str{ii};
    tmp_str_test = test_loss_str{ii};
    
    index = find(tmp_str_test=='@');
    loss_iter(ii) = str2double(tmp_str_test(1:index-1));
%     ortrain_loss(ii) = str2double(tmp_str_ortrain(index+1:end));
    
%     index = find(tmp_str_ver=='@');
%     ver_loss(ii) = str2double(tmp_str_ver(index+1:end));
    
    index = find(tmp_str_test=='@');
    test_loss(ii) = str2double(tmp_str_test(index+1:end));
    
end

% ����loss���ߣ��ֱ��ǱȽ�ԭʼ�ĺ;���ƽ����
figure;
subplot(2,1,1);
plot(1:length(smooth(smooth(minibatch_loss,5))),smooth(smooth(minibatch_loss,5))/batch_size,'k');hold on;
% plot(loss_iter,(ver_loss),'r');hold on;
plot(loss_iter,(test_loss),'g');
% plot(loss_iter,(ortrain_loss),'b');
xlabel('iter');
ylabel('loss');
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
legend({'minibatch loss','test loss'},'Location','best');
subplot(2,1,2);
plot(1:length(smooth(smooth(minibatch_loss,50))),smooth(smooth(minibatch_loss,50))/batch_size,'k');hold on;
% plot(loss_iter,smooth(ver_loss,50),'r');hold on;
plot(loss_iter,smooth(test_loss,50),'g','LineWidth',1);
% plot(loss_iter,smooth(ortrain_loss,50),'b');
xlabel('iter');
ylabel('loss');
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
legend({'minibatch loss','test loss trend'},'Location','best');

%% step3����ȡverify����result���
% if exchange_flag == 0
%     tmp_str_v = '_ver_result';
% else
%     tmp_str_v = '_test_result';
% end
% 
% file_name = strcat(log_path,filesep,'@',fold_name,tmp_str_v,'.txt');
% ver_result_str = importdata(file_name);
% ver_result = zeros(length(ver_result_str),6);
% for ii = 1:length(ver_result_str)
%     tmp_str = ver_result_str{ii};
%     %iter
%     index = find(tmp_str=='@');
%     iter = str2double(tmp_str(1:index-1));
%     tmp_str = tmp_str(index+2:end);
%     %acc
%     index = find(tmp_str==',');
%     acc = str2double(tmp_str(1:index(1)-1));
%     tmp_str = tmp_str(index(1)+1:end);
%     %sen
%     index = find(tmp_str==',');
%     sen = str2double(tmp_str(1:index(1)-1));
%     tmp_str = tmp_str(index(1)+1:end);
%     %spc
%     index = find(tmp_str==',');
%     spc = str2double(tmp_str(1:index(1)-1));
%     tmp_str = tmp_str(index(1)+1:end);
%     %auc
%     index = find(tmp_str==',');
%     auc = str2double(tmp_str(1:index(1)-1));
%     tmp_str = tmp_str(index(1)+1:end);
%     %loss
%     index = find(tmp_str==']');
%     loss = str2double(tmp_str(1:index(1)-1));
%     
%     
%     ver_result(ii,:) = [iter,acc,sen,spc,auc,loss];  %========================
% end
% 

%% step4����ȡtest����result���
if exchange_flag == 0
    tmp_str_v = '_test_result';
else
    tmp_str_v = '_ver_result';
end

file_name = strcat(log_path,filesep,'@',fold_name,tmp_str_v,'.txt');
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


%% step5����ݽ�ȡ��
% ���ǵ���������ݱ��治ȫ��������������е�ʱ��ĳʱ���жϣ�����ȡmin�������
% cut_index = min(length(test_result_str),length(ver_result_str));
% if length(test_result_str) == cut_index
%     ver_result = ver_result(1:cut_index,:);
% else
%     test_result = test_result(1:cut_index,:);
% end
%% step6��������ָ�꣺��������ѡģ�͵�ָ�꣩
%����ָ�꣺======================================
%--�� 1 ָ�꣺acc
%--�� 2 ָ�꣺sen
%--�� 3 ָ�꣺spc
%--�� 4 ָ�꣺auc
%--�� 5 ָ�꣺loss
%��ָ��=============================================
%���±�7��ʼ����Ϊԭ�����һ����iters
%--�� 6 ָ�꣺��acc+sen+spc+auc��/loss   �����ָ����ͺ󣬳���loss
%--�� 7 ָ�꣺��acc+sen+spc+auc��+ a/loss ��a��������Լ�����Խ���ʾloss�Ĺ���Խ��
%--�� 8 ָ�꣺��acc+sen+spc+auc��
%--�� 9 ָ�꣺��acc*sen*spc*auc��/loss
%--��10 ָ�꣺��acc*sen*spc*auc��
%--��11 ָ�꣺��sen+spc��
%--��12 ָ�꣺��sen+spc��/loss
%--��13 ָ�꣺��sen*spc��
%--��14 ָ�꣺��sen*spc��/loss
%--��15 ָ�꣺sen+auc

%��6ָ��
% ver_result(:,7)=(ver_result(:,2)+ver_result(:,3)+ ver_result(:,4)+ver_result(:,5))./ver_result(:,6);
test_result(:,7)=(test_result(:,2)+test_result(:,3)+ test_result(:,4)+test_result(:,5))./test_result(:,6);
%��7ָ��
% ver_result(:,8)=(ver_result(:,2)+ver_result(:,3)+ ver_result(:,4)+ver_result(:,5))+a./ver_result(:,6);
test_result(:,8)=(test_result(:,2)+test_result(:,3)+ test_result(:,4)+test_result(:,5))+a./test_result(:,6);
%��8ָ��
% ver_result(:,9)=(ver_result(:,2)+ver_result(:,3)+ ver_result(:,4)+ver_result(:,5));
test_result(:,9)=(test_result(:,2)+test_result(:,3)+ test_result(:,4)+test_result(:,5));
%��9ָ��
% ver_result(:,10)=(ver_result(:,2).*ver_result(:,3).* ver_result(:,4).*ver_result(:,5))./ver_result(:,6);
test_result(:,10)=(test_result(:,2).*test_result(:,3).* test_result(:,4).*test_result(:,5))./test_result(:,6);
%��10ָ��
% ver_result(:,11)=(ver_result(:,2).*ver_result(:,3).* ver_result(:,4).*ver_result(:,5));
test_result(:,11)=(test_result(:,2).*test_result(:,3).* test_result(:,4).*test_result(:,5));
%��11ָ��
% ver_result(:,12)=(ver_result(:,3)+ ver_result(:,4));
test_result(:,12)=(test_result(:,3)+ test_result(:,4));
%��12ָ��
% ver_result(:,13)=(ver_result(:,3)+ ver_result(:,4))./ver_result(:,6);
test_result(:,13)=(test_result(:,3)+ test_result(:,4))./test_result(:,6);
%��13ָ��
% ver_result(:,14)=(ver_result(:,3).* ver_result(:,4));
test_result(:,14)=(test_result(:,3).* test_result(:,4));
%��14ָ��
% ver_result(:,15)=(ver_result(:,3).* ver_result(:,4))./ver_result(:,6);
test_result(:,15)=(test_result(:,3).* test_result(:,4))./test_result(:,6);
%��15ָ��
% ver_result(:,16)=(ver_result(:,3)+ ver_result(:,5));
test_result(:,16)=(test_result(:,3)+ test_result(:,5));


%% xianshiyixia
% figure;
% subplot(5,1,1);
% plot(ver_result(:,1),ver_result(:,2),'b');hold on;
% plot(ver_result(:,1),smooth(ver_result(:,2),20),'r');
% title('ver acc');
% subplot(5,1,2);
% plot(ver_result(:,1),ver_result(:,3),'b');hold on;
% plot(ver_result(:,1),smooth(ver_result(:,3),20),'r');
% title('sen');
% subplot(5,1,3);
% plot(ver_result(:,1),ver_result(:,4),'b');hold on;
% plot(ver_result(:,1),smooth(ver_result(:,4),20),'r');
% title('spc');
% subplot(5,1,4);
% plot(ver_result(:,1),ver_result(:,5),'b');hold on;
% plot(ver_result(:,1),smooth(ver_result(:,5),20),'r');
% title('auc');
% subplot(5,1,5);
% plot(ver_result(:,1),ver_result(:,6),'b');hold on;
% plot(ver_result(:,1),smooth(ver_result(:,6),20),'r');
% title('sum');
% figure;
% plot(ver_result(:,1),ver_result(:,2).*ver_result(:,3).*ver_result(:,4).*ver_result(:,5),'b');hold on;
% plot(ver_result(:,1),smooth(ver_result(:,2).*ver_result(:,3).*ver_result(:,4).*ver_result(:,5),20),'r');
% title('ver miul');

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
% ���յ�indexxָ�����򲢽�ȡ��ȡ��Ϊallv
% ver_result = ver_result(ver_result(:,1)>threshold_iter,:);%���Ȱ�С��һ�������Ŀ���
% 
% % ����ȡ���򣬷���鿴
% ver_result = sortrows(ver_result,indexx+1);
% ver_result = ver_result(end:-1:1,:);
% 
% % ��Ϊ��ͬ������indexx��ָ���ֵ���ܻ���ͬ��������ʱ�Ȱѳ��ֹ��ֵȡ����
% allv_value = [];
% for ii = 1:size(ver_result,1)
%     if ~(ismember(ver_result(ii,indexx+1),allv_value))
%         allv_value = [allv_value,ver_result(ii,indexx+1)];
%     end
% end
% allv_value = sort(allv_value,'descend');
% allv_value = allv_value';
% 
% % ֮������Щֵ���Ӹߵ��ͣ��Ѷ�Ӧiter��ver��result������
% % ϸ�ڣ�ÿ��ֵ��Ӧ������iterȫ����������ÿ���һ��ֵ����һ����iter��result���͸���һ����������iter����
% % һ���������Ԥ��� ���ɶ� �� ��ֹͣ��ѡ
% allv_select_num = 0;%����accɸѡ����֤������ļ�������
% for ii = 1:length(allv_value)
%     tmp_index =  find(ver_result(:,indexx+1)==allv_value(ii));
%     allv_select_num = allv_select_num+length(tmp_index);
%     if allv_select_num >= min_test_select_num
%         break;
%     end
% end
% 
% % ֮���ver��result��ȡ����ʱ�����ַ���
% % ��һ���£���Ϊһ��ָ��ͬһ������ܶ�Ӧ�ܶ��ļ�������ȡ�಻ȥ��(Ҳ���ǿ��ɶȸ�һ�㣡)����Ϊadapt ģʽ
% result_by_allv = ver_result(1:allv_select_num,:);
% % �������£�ֱ����Ԥ�����С��ȡ��Ҳ�������ɶȣ���ȡ���Ƚ��ϸ񣬵��޸��
% %result_by_allv = ver_result_sort_by_allv(1:min_test_select_num,:);%ֱ��ȡ��Сֵ
% 
% %% step8������ɸѡ���iter����ȡtest���ϵ�����ָ��
% 
% 
% 
% ilter_final = result_by_allv(:,1);
% 
% % �����֤��ָ��ɸѡ�����ĵ�����ȡ���Լ���result��֮������ѡ��һ����õģ�
% final_test_result = [];
% for ii = 1:length(ilter_final)
%     index = find(test_result(:,1)==ilter_final(ii));
%     final_test_result = [final_test_result;test_result(index,:)];
% end
% 
% % ��ɸѡ�����Ĳ��Լ��Ÿ���
% final_test_result = sortrows(final_test_result,indexx+1);
% final_test_result = final_test_result(end:-1:1,:);
% 
% % �����һ��ֵ����������Ҫ��test���
% final_value = final_test_result(1,:);
% 
% % ȥ���Լ��ֶ���ӵ�ָ�꣬���ҽ�����������a��ͷcopyΪ�µı���������鿴
% final_value = final_value(1:6);
% a_final_value = final_value;
% a_iters = final_value(1);
% a_acc = final_value(2);
% a_sen = final_value(3);
% a_spc = final_value(4);
% a_AUC = final_value(5);
% a_Loss = final_value(6);
% 
% %% �õ������ս��֮������Ҫ�Ѷ�Ӧ������pre��label����idȡ����
% if exchange_flag == 0
%     tmp_str_s = 'test';
% else
%     tmp_str_s = 'ver';
% end
% 
% % ���ȶ�ȡid
% test_id_path = strcat(log_path,filesep,'@',fold_name,'_',tmp_str_s,'_id.txt');
% test_id = importdata(test_id_path);
% test_id = test_id{1};
% index_ate = find(test_id=='@');
% test_id = test_id(index_ate+2:end-1);
% test_ID = strsplit(test_id,',');
% test_ID = test_ID';
% test_ID = str2double(test_ID);
% 
% % Ȼ���ȡlabel
% test_label_path = strcat(log_path,filesep,'@',fold_name,'_',tmp_str_s,'_label.txt');
% test_label = importdata(test_label_path);
% test_label = test_label{1};
% index_ate = find(test_label=='@');
% test_label = test_label(index_ate+2:end-1);
% test_label = strsplit(test_label,',');
% test_label = test_label';
% test_label = str2double(test_label);
% 
% 
% 
% % ����ȡpre_value
% test_pre_path = strcat(log_path,filesep,'@',fold_name,'_',tmp_str_s,'_pre.txt');
% test_pre = importdata(test_pre_path);
% for i = 1:length(test_pre)
%     index_ate = find(test_pre{i}=='@');
%     iterrrr = str2double(test_pre{i}(1:index_ate-1));
%     if iterrrr == final_value(1)
%         test_prevalue = test_pre{i}(index_ate+2:end-1);
%         test_prevalue = strsplit(test_prevalue,',');
%         test_prevalue = test_prevalue';
%         test_prevalue = str2double(test_prevalue);
%         break;
%     end    
% end
% 
% 
% %% save
% save(strcat(save_path,filesep,'folder_',fold_name),'test_label','test_prevalue','test_ID','final_value');

% pause(10)
% end