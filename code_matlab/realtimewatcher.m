%% step1��initialization
clear;
clc;
close all
%% step2��set the log path and other parameter
log_path = 'F:\log';%
fold_name = '1';
or_train_flag = 1;
verify_flag = 1;
test_flag = 1;
%% show the LRate
lr_path = strcat(log_path,filesep,'@',fold_name,'_lr.txt');
lr =  importdata(lr_path);
figure;
plot(lr,'b');
legend({'lr'},'Location','best');
%% show the minibatch_loss
minibatch_loss_path = strcat(log_path,filesep,'@',fold_name,'_loss.txt');
minibatch_loss =  importdata(minibatch_loss_path);
figure;
plot(minibatch_loss,'b');
hold on;
plot(smooth(minibatch_loss),'r');
hold on;
plot(smooth(smooth(minibatch_loss,50)),'g');
hold on;
plot(smooth(smooth(minibatch_loss,800)),'m');
legend({'mini loss','smooth level1','smooth level2','smooth level3'},'Location','best');
%% show the test & verify & or_train & minibatch_train loss
legend_str = {};
legend_str{end+1}='minibatch';
plot_color = {'r','g','b'};
all_loss = {};
all_iter = {};


if or_train_flag
    legend_str{end+1}='or train';
    tmp_str_file = '_or_loss';
    loss_path = strcat(log_path,filesep,'@',fold_name,tmp_str_file,'.txt');
    tmp_loss_str = importdata(loss_path);
    tmp_loss = [];
    tmp_iter = [];
    for ii= 1:length(tmp_loss_str)
        tmp_str = tmp_loss_str{ii};
        index = find(tmp_str=='@');    
        tmp_loss(ii) = str2double(tmp_str(index+1:end));
        tmp_iter(ii) = str2double(tmp_str(1:index-1));
    end
    all_loss{end+1}=tmp_loss;
    all_iter{end+1}=tmp_iter;
end

if verify_flag
    legend_str{end+1}='verify';
    tmp_str_file = '_ver_loss';
    loss_path = strcat(log_path,filesep,'@',fold_name,tmp_str_file,'.txt');
    tmp_loss_str = importdata(loss_path);
    tmp_loss = [];
    tmp_iter = [];
    for ii= 1:length(tmp_loss_str)
        tmp_str = tmp_loss_str{ii};
        index = find(tmp_str=='@');    
        tmp_loss(ii) = str2double(tmp_str(index+1:end));
        tmp_iter(ii) = str2double(tmp_str(1:index-1));
    end
    all_loss{end+1}=tmp_loss;
    all_iter{end+1}=tmp_iter;
end


if test_flag
    legend_str{end+1}='test';
    tmp_str_file = '_test_loss';
    loss_path = strcat(log_path,filesep,'@',fold_name,tmp_str_file,'.txt');
    tmp_loss_str = importdata(loss_path);
    tmp_loss = [];
    tmp_iter = [];
    for ii= 1:length(tmp_loss_str)
        tmp_str = tmp_loss_str{ii};
        index = find(tmp_str=='@');    
        tmp_loss(ii) = str2double(tmp_str(index+1:end));
        tmp_iter(ii) = str2double(tmp_str(1:index-1));
    end
    all_loss{end+1}=tmp_loss;
    all_iter{end+1}=tmp_iter;
end

%show loss by loop
figure;
subplot(2,1,1);
plot(minibatch_loss,'k');hold on;
for i = 1:length(all_loss)
    plot(all_iter{i},all_loss{i},plot_color{i},'LineWidth',1);
    hold on;
end
xlabel('iter');
ylabel('loss');
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
legend(legend_str,'Location','best');
title('or loss without smooth');

subplot(2,1,2);
plot(smooth(smooth(minibatch_loss,500)),'k');hold on;
for i = 1:length(all_loss)
    plot(all_iter{i},smooth(all_loss{i},50),plot_color{i},'LineWidth',1);
    hold on;
end
xlabel('iter');
ylabel('loss');
backColor = [0.7 0.7 0.4];set(gca, 'color', backColor);
legend(legend_str,'Location','best');
title('or loss with smooth');

%% step3����ȡverify����result���?
if verify_flag
    tmp_str_v = '_ver_result';
    
    
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
        
        
        ver_result(ii,:) = [iter,acc,sen,spc,auc,loss];
    end
    
    
    
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
%     figure;
%     plot(ver_result(:,1),ver_result(:,2).*ver_result(:,3).*ver_result(:,4).*ver_result(:,5),'b');hold on;
%     plot(ver_result(:,1),smooth(ver_result(:,2).*ver_result(:,3).*ver_result(:,4).*ver_result(:,5),20),'r');
%     title('ver miul');
    
end
%% step4����ȡtest����result���?
if test_flag
    tmp_str_v = '_test_result';
    
    
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
%     figure;
%     plot(test_result(:,1),test_result(:,2).*test_result(:,3).*test_result(:,4).*test_result(:,5),'b');hold on;
%     plot(test_result(:,1),smooth(test_result(:,2).*test_result(:,3).*test_result(:,4).*test_result(:,5),20),'r');
%     title('test miul');
end























