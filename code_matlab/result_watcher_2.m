%% 绘图：loss

% plot 部分
% =========================================================================
% 显示所有病人平均的loss，以及病人平均的loss，以及block直接平均的loss（即不按照病人加权平均）（直接plot即可）
figure;
subplot(3,1,1);
plot(all_loss_iter, all_loss,'r','LineWidth',2);hold on;
plot(person_pre_iter,mean(person_loss_all,1),'g');hold on;
plot(person_pre_iter,mean(block_loss,1),'b');
legend({'mean loss','mean per loss','mean block loss'},'Location','best');
title('loss')

% 显示所有病人各自的loss
subplot(3,1,2);
for ii = 1:length(person_id)
   plot(person_pre_iter,person_loss_all(ii,:)) ; hold on;
end
hold off;
title('all person loss');

% 显示所有block的loss
subplot(3,1,3);
for ii = 1:length(per_block_name)
   plot(person_pre_iter,block_loss(ii,:)) ; hold on;
end
hold off;
title('all block loss');


% heatmap 部分
% =========================================================================
% person ============================
figure;
imagesc(imresize(person_loss_all,size(person_loss_all)*scale_size,'nearest'));
colormap(parula);
colorbar('location','SouthOutside');
ylabel('subject id');
xlabel('iters');

ytick = floor(1:(size(person_loss_all,1)*scale_size/length(person_id)):size(person_loss_all,1)*scale_size);
set(gca, 'yTick', ytick);
set(gca,'YTickLabel',person_id)

xtick = floor(1:(size(person_loss_all,2)*scale_size/length(all_loss_iter)):size(person_loss_all,2)*scale_size);
set(gca, 'xTick', xtick);
set(gca,'XTickLabel',all_loss_iter)

title('person loss heatmap');

% block ============================
figure;
imagesc(imresize(block_loss,size(block_loss)*scale_size,'nearest'));
colormap(parula);
colorbar('location','SouthOutside');
ylabel('block name');
xlabel('iters');

ytick = floor(1:(size(block_loss,1)*scale_size/length(per_block_name)):size(block_loss,1)*scale_size);
set(gca, 'yTick', ytick);
set(gca,'YTickLabel',per_block_name)

xtick = floor(1:(size(block_loss,2)*scale_size/length(all_loss_iter)):size(block_loss,2)*scale_size);
set(gca, 'xTick', xtick);
set(gca,'XTickLabel',all_loss_iter)

title('block loss heatmap');




%% 绘图：pre
% block
figure;
imagesc(imresize(block_pre,size(block_pre)*scale_size,'nearest'));
colormap(parula);
colorbar('location','SouthOutside');
ylabel('id');
xlabel('iters');

ytick = floor(1:(size(block_pre,1)*scale_size/length(per_block_name)):size(block_pre,1)*scale_size);
set(gca, 'yTick', ytick);
set(gca,'YTickLabel',per_block_name)

xtick = floor(1:(size(block_pre,2)*scale_size/length(all_loss_iter)):size(block_pre,2)*scale_size);
set(gca, 'xTick', xtick);
set(gca,'XTickLabel',all_loss_iter)

title('block pre heatmap');

% person
figure;
imagesc(imresize(person_pre_all,size(person_pre_all)*scale_size,'nearest'));
colormap(parula);
colorbar('location','SouthOutside');
ylabel('id');
xlabel('iters');

ytick = floor(1:(size(person_pre_all,1)*scale_size/length(person_id)):size(person_pre_all,1)*scale_size);
set(gca, 'yTick', ytick);
set(gca,'YTickLabel',person_id)

xtick = floor(1:(size(person_pre_all,2)*scale_size/length(all_loss_iter)):size(person_pre_all,2)*scale_size);
set(gca, 'xTick', xtick);
set(gca,'XTickLabel',all_loss_iter)

title('person pre heatmap');


%% acc
% person的acc有两种显示方式，一种是只看单个person的最终pre，之后做成二值图
% 另外一种是看person的所有block有哪些预测对了
person_right = double(person_pre_label == person_true_label);
figure;
imagesc(imresize(person_right,size(person_right)*scale_size,'nearest'));
colormap(parula);
colorbar('location','SouthOutside');
ylabel('id');
xlabel('iters');

ytick = floor(1:(size(person_right,1)*scale_size/length(person_id)):size(person_right,1)*scale_size);
set(gca, 'yTick', ytick);
set(gca,'YTickLabel',person_id)

xtick = floor(1:(size(person_right,2)*scale_size/length(all_loss_iter)):size(person_right,2)*scale_size);
set(gca, 'xTick', xtick);
set(gca,'XTickLabel',all_loss_iter)

title('person right heatmap, hot is right');



%% 每个病人分别有多少块预测对
person_block_acc = [];
for i = 1:length(person_id)
    id = str2double(person_id{i});
    tmp_pre_label = block_pre_label(block_name(:,1) == id,:);
    tmp_true_label = block_true_label(block_name(:,1) == id,:);
    tmp_right = tmp_pre_label == tmp_true_label;
    tmp_acc = sum(tmp_right,1)/size(tmp_right,1);
    person_block_acc = [person_block_acc;tmp_acc];
end
figure;
imagesc(imresize(person_block_acc,size(person_block_acc)*scale_size,'nearest'));
colormap(parula);
colorbar('location','SouthOutside');
ylabel('id');
xlabel('iters');

ytick = floor(1:(size(person_block_acc,1)*scale_size/length(person_id)):size(person_block_acc,1)*scale_size);
set(gca, 'yTick', ytick);
set(gca,'YTickLabel',person_id)

xtick = floor(1:(size(person_block_acc,2)*scale_size/length(all_loss_iter)):size(person_block_acc,2)*scale_size);
set(gca, 'xTick', xtick);
set(gca,'XTickLabel',all_loss_iter)

title('person acc (right block / all block) heatmap');







%% 删除一些变量，最后只留下block相关的即可
clear all_loss logpath loss_all 



