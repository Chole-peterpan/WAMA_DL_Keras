%% 需要看的指标
% allv_select_num 经过筛选后最终留下的验证集数量
%

%% step1：初始化
clear;
clc;
close all
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
%第15指标：sen+auc


save_path = 'G:\git\asd';
log_path = 'G:\git\fold5\log';%
fold_name = '5';
a = 2;
indexx = 8;%使用指标的序号，1就是第1指标,最终会按照这个指标来选择权重，指标越大表示模型越好（除了loss越小表示模型越好）
batch_size = 1;
% finf = dir([fold_path,'\*.txt']);
min_test_select_num = 40;%最终挑选的测试集的结果文件的最小数量（即不同iteration的数量）
threshold_iter = 39999;%因前n次迭代的模型师不可靠的，所以需要手动设置迭代数阈值把前n次的结果卡掉
exchange_flag = 0;%1则test与ver集合互换
%% 显示学习率曲线
lr_path = strcat(log_path,filesep,'@',fold_name,'_lr.txt');
lr =  importdata(lr_path);
figure;
plot(lr,'b');
legend({'lr'},'Location','best');
%% 显示一下minibatch_loss

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
%% 显示4个loss：叠加显示
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

% 绘制loss曲线，分别是比较原始的和经过平滑的
figure;
subplot(2,1,1);
plot(1:length(smooth(smooth(minibatch_loss,5))),smooth(smooth(minibatch_loss,5))/batch_size,'k');hold on;
plot(loss_iter,(ver_loss),'r');hold on;
plot(loss_iter,(test_loss),'g');hold on;
plot(loss_iter,(ortrain_loss),'b');
xlabel('iter');
ylabel('loss');
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
legend({'minibatch loss','ver loss','test loss','ortrain loss'},'Location','best');
subplot(2,1,2);
plot(1:length(smooth(smooth(minibatch_loss,50))),smooth(smooth(minibatch_loss,50))/batch_size,'k');hold on;
plot(loss_iter,smooth(ver_loss,50),'r');hold on;
plot(loss_iter,smooth(test_loss,50),'g','LineWidth',1);hold on;
plot(loss_iter,smooth(ortrain_loss,50),'b');
xlabel('iter');
ylabel('loss');
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
legend({'minibatch loss','ver loss trend','test loss trend','ortrain loss trend'},'Location','best');

%% step3：读取verify的总result数据
if exchange_flag == 0
    tmp_str_v = '_ver_result';
else
    tmp_str_v = '_test_result';
end

file_name = strcat(log_path,filesep,'@',fold_name,tmp_str_v,'.txt');
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


%% step4：读取test的总result数据
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


%% step5：数据截取：
% 考虑到可能有数据保存不全的情况，比如运行的时候某时刻中断，所以取min迭代数量
cut_index = min(length(test_result_str),length(ver_result_str));
if length(test_result_str) == cut_index
    ver_result = ver_result(1:cut_index,:);
else
    test_result = test_result(1:cut_index,:);
end
%% step6：构建新指标：（用来挑选模型的指标）
%已有指标：======================================
%--第 1 指标：acc
%--第 2 指标：sen
%--第 3 指标：spc
%--第 4 指标：auc
%--第 5 指标：loss
%新指标=============================================
%从下标7开始，因为原矩阵第一列是iters
%--第 6 指标：（acc+sen+spc+auc）/loss   正相关指标求和后，除以loss
%--第 7 指标：（acc+sen+spc+auc）+ a/loss （a这个参数自己定，越大表示loss的贡献越大）
%--第 8 指标：（acc+sen+spc+auc）
%--第 9 指标：（acc*sen*spc*auc）/loss
%--第10 指标：（acc*sen*spc*auc）
%--第11 指标：（sen+spc）
%--第12 指标：（sen+spc）/loss
%--第13 指标：（sen*spc）
%--第14 指标：（sen*spc）/loss
%--第15 指标：sen+auc

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
%第15指标
ver_result(:,16)=(ver_result(:,3)+ ver_result(:,5));
test_result(:,16)=(test_result(:,3)+ test_result(:,5));


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
% 按照第indexx指标排序并截取：取名为allv
ver_result = ver_result(ver_result(:,1)>threshold_iter,:);%首先把小于一定迭代数的卡掉

% 排序并取倒序，方便查看
ver_result = sortrows(ver_result,indexx+1);
ver_result = ver_result(end:-1:1,:);

% 因为不同迭代数，第indexx个指标的值可能会相同，所以暂时先把出现过的值取出来
allv_value = [];
for ii = 1:size(ver_result,1)
    if ~(ismember(ver_result(ii,indexx+1),allv_value))
        allv_value = [allv_value,ver_result(ii,indexx+1)];
    end
end
allv_value = sort(allv_value,'descend');
allv_value = allv_value';

% 之后按照这些值，从高到低，把对应iter的ver的result挑出来
% 细节：每个值对应的所有iter全都挑出来，每根据一个值挑出一部分iter的result，就更新一次挑出来的iter总数
% 一旦总数大于预设的 宽松度 ， 则停止挑选
allv_select_num = 0;%经过acc筛选的验证集结果文件的数量
for ii = 1:length(allv_value)
    tmp_index =  find(ver_result(:,indexx+1)==allv_value(ii));
    allv_select_num = allv_select_num+length(tmp_index);
    if allv_select_num >= min_test_select_num
        break;
    end
end

% 之后对ver的result截取，此时有两种方法
% 法一如下，因为一个指标同一个数可能对应很多文件，尽量取多不去少(也就是宽松度高一点！)，成为adapt 模式
result_by_allv = ver_result(1:allv_select_num,:);
% 法二如下，直接以预设的最小截取数（也叫作宽松度）截取，比较严格，但无根据
%result_by_allv = ver_result_sort_by_allv(1:min_test_select_num,:);%直接取最小值

%% step8：按照筛选后的iter，提取test集合的所有指标



ilter_final = result_by_allv(:,1);

% 根据验证集指标筛选出来的迭代数，截取测试集的result（之后会从中选出一个最好的）
final_test_result = [];
for ii = 1:length(ilter_final)
    index = find(test_result(:,1)==ilter_final(ii));
    final_test_result = [final_test_result;test_result(index,:)];
end

% 给筛选出来的测试集排个序
final_test_result = sortrows(final_test_result,indexx+1);
final_test_result = final_test_result(end:-1:1,:);

% 排序第一的值，就是最终要的test结果
final_value = final_test_result(1,:);

% 去掉自己手动添加的指标，并且将各个分量以a开头copy为新的变量，方便查看
final_value = final_value(1:6);
a_final_value = final_value;
a_iters = final_value(1);
a_acc = final_value(2);
a_sen = final_value(3);
a_spc = final_value(4);
a_AUC = final_value(5);
a_Loss = final_value(6);

%% 得到了最终结果之后，我们要把对应迭代数的pre和label还有id取出来
if exchange_flag == 0
    tmp_str_s = 'test';
else
    tmp_str_s = 'ver';
end

% 首先读取id
test_id_path = strcat(log_path,filesep,'@',fold_name,'_',tmp_str_s,'_id.txt');
test_id = importdata(test_id_path);
test_id = test_id{1};
index_ate = find(test_id=='@');
test_id = test_id(index_ate+2:end-1);
test_ID = strsplit(test_id,',');
test_ID = test_ID';
test_ID = str2double(test_ID);

% 然后读取label
test_label_path = strcat(log_path,filesep,'@',fold_name,'_',tmp_str_s,'_label.txt');
test_label = importdata(test_label_path);
test_label = test_label{1};
index_ate = find(test_label=='@');
test_label = test_label(index_ate+2:end-1);
test_label = strsplit(test_label,',');
test_label = test_label';
test_label = str2double(test_label);



% 最后读取pre_value
test_pre_path = strcat(log_path,filesep,'@',fold_name,'_',tmp_str_s,'_pre.txt');
test_pre = importdata(test_pre_path);
for i = 1:length(test_pre)
    index_ate = find(test_pre{i}=='@');
    iterrrr = str2double(test_pre{i}(1:index_ate-1));
    if iterrrr == final_value(1)
        test_prevalue = test_pre{i}(index_ate+2:end-1);
        test_prevalue = strsplit(test_prevalue,',');
        test_prevalue = test_prevalue';
        test_prevalue = str2double(test_prevalue);
        break;
    end    
end


%% save
save(strcat(save_path,filesep,'folder_',fold_name),'test_label','test_prevalue','test_ID','final_value');
