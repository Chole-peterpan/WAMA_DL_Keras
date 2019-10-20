%% 需要看的指标
% allv_select_num 经过筛选后最终留下的验证集数量
% 

%% step1：初始化
clear;
clc;

%% step2：参数和路径设置
%第1指标：acc
%第2指标：sen
%第3指标：spc
%第4指标：auc
%第5指标：loss
%第6指标：（acc+sen+spc+auc）/loss   正相关指标求和后，除以loss
%第7指标：（acc+sen+spc+auc）+ a/loss （a这个参数自己定，越大表示loss的贡献越大）
%第8指标：（acc+sen+spc+auc）
%第9指标：（acc*sen*spc*auc）/loss 
%第10指标：（acc*sen*spc*auc）
%第11指标：（sen+spc）
%第12指标：（sen+spc）/loss
%第13指标：（sen*spc）
%第14指标：（sen*spc）/loss

save_path = 'G:\qweqweqweqwe';
fold_path = 'G:\result\test4';%
fold_name = '2';
a = 1;
indexx = 1;%使用指标的序号，1就是第1指标,最终会按照这个指标来选择权重，指标越大表示模型越好（除了loss越小表示模型越好）
batch_size = 6;
finf = dir([fold_path,'\*.txt']);
min_test_select_num = 40;%最终挑选的测试集的结果文件的最小数量（即不同iteration的数量）

%% 显示一下minibatch_loss

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
%% 显示3个loss：叠加显示
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

%考虑到可能有数据保存不全的情况，比如运行的时候某时刻中断，所以取min迭代数量
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
legend({'ver loss','test loss','ortrain loss','minibatch loss','ver loss trend','test loss trend','ortrain loss trend'},'Location','best'); %依据绘图的先后顺序，一次标注曲线。
%% step3：读取verify的总数据
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


%% step4：读取test的总数据
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

%% step5：数据截取：    为啥数据保存不全呢？？？？ fuck
cut_index = min(length(test_result_str),length(ver_result_str));
if length(test_result_str) == cut_index
    ver_result = ver_result(1:cut_index,:);
else
    test_result = test_result(1:cut_index,:);
end
%% step6：构建新指标：（用来挑选模型的指标）
%已有指标：======================================
%第1指标：acc
%第2指标：sen
%第3指标：spc
%第4指标：auc
%第5指标：loss
%新指标：从下表7开始，因为原矩阵第一列是iters===========================================
%第6指标：（acc+sen+spc+auc）/loss   正相关指标求和后，除以loss
%第7指标：（acc+sen+spc+auc）+ a/loss （a这个参数自己定，越大表示loss的贡献越大）
%第8指标：（acc+sen+spc+auc）
%第9指标：（acc*sen*spc*auc）/loss 
%第10指标：（acc*sen*spc*auc）
%第11指标：（sen+spc）
%第12指标：（sen+spc）/loss
%第13指标：（sen*spc）
%第14指标：（sen*spc）/loss

%第6指标
ver_result(:,7)=(ver_result(:,2)+ver_result(:,3)+ ver_result(:,4)+ver_result(:,5))./ver_result(:,6);
test_result(:,7)=(test_result(:,2)+test_result(:,3)+ test_result(:,4)+test_result(:,5))./test_result(:,6);
%第7指标
ver_result(:,8)=(ver_result(:,2)+ver_result(:,3)+ ver_result(:,4)+ver_result(:,5))+a./ver_result(:,6);
test_result(:,8)=(test_result(:,2)+test_result(:,3)+ test_result(:,4)+test_result(:,5))+a./test_result(:,6);
%第8指标
ver_result(:,9)=(ver_result(:,2)+ver_result(:,3)+ ver_result(:,4)+ver_result(:,5));
test_result(:,9)=(test_result(:,2)+test_result(:,3)+ test_result(:,4)+test_result(:,5));
%第9指标
ver_result(:,10)=(ver_result(:,2).*ver_result(:,3).* ver_result(:,4).*ver_result(:,5))./ver_result(:,6);
test_result(:,10)=(test_result(:,2).*test_result(:,3).* test_result(:,4).*test_result(:,5))./test_result(:,6);
%第10指标
ver_result(:,11)=(ver_result(:,2).*ver_result(:,3).* ver_result(:,4).*ver_result(:,5));
test_result(:,11)=(test_result(:,2).*test_result(:,3).* test_result(:,4).*test_result(:,5));  
%第11指标
ver_result(:,12)=(ver_result(:,3)+ ver_result(:,4));
test_result(:,12)=(test_result(:,3)+ test_result(:,4));  
%第12指标
ver_result(:,13)=(ver_result(:,3)+ ver_result(:,4))./ver_result(:,6);
test_result(:,13)=(test_result(:,3)+ test_result(:,4))./test_result(:,6); 
%第13指标
ver_result(:,14)=(ver_result(:,3).* ver_result(:,4));
test_result(:,14)=(test_result(:,3).* test_result(:,4));  
%第14指标
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
%% step7：按照第indexx指标筛选最终iter
%按照第indexx指标排序并截取：取名为allv
%==========================================！！！！！！！！！！！！！！！
ver_result_sort_by_allv = sortrows(ver_result,indexx+1);
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
%因为一个指标同一个数可能对应很多文件，尽量取多不去少(上面那一句这种更科学啊！)，成为adapt 模式
%result_by_allv = ver_result_sort_by_allv(1:min_test_select_num,:);%直接取最小值
%按照第5指标排序并截取：=====================================================


%% step8：按照筛选后的iter，提取test集合的所有指标
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
a_final_value = final_value;%以a开头copy以下，方便查看。
a_iters = final_value(1);
a_acc = final_value(2);
a_sen = final_value(3);
a_spc = final_value(4);
a_AUC = final_value(5);
a_Loss = final_value(6);
%% save
save(strcat(save_path,filesep,'folder_',fold_name),'test_label','test_prevalue','test_ID','final_value'); 
