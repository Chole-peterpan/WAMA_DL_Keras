clear;clc;
%% load 
filepathh = 'G:\git\asd';
% filepathh = 'H:\diploma_result\result\@以acc+sen+spc+auc位置表，迭代数严格限制\resnet_strict';
folder_num = 5;%折数



for ii= 1:folder_num
    tmp_path = strcat(filepathh,filesep,'folder_',num2str(ii),'.mat');
    folder(ii) = load(tmp_path);
end
label = [];
predict = [];
ID = [];
for ii = 1:folder_num
    label = [label;folder(ii).test_label];
    predict = [predict;folder(ii).test_prevalue];
    ID = [ID;folder(ii).test_ID];
end

% 手动滑稽版本-------------------------------------------------------------
% folder(1) = load(strcat(filepathh,filesep,'folder_1.mat'));
% folder(2) = load(strcat(filepathh,filesep,'folder_2.mat'));
% folder(3) =  load(strcat(filepathh,filesep,'folder_3.mat'));
% folder(4) =  load(strcat(filepathh,filesep,'folder_4.mat'));
% folder(5) =  load(strcat(filepathh,filesep,'folder_5.mat'));
% label = [fold1.test_label;fold2.test_label;fold3.test_label;fold4.test_label;fold5.test_label];
% predict = [fold1.test_prevalue;fold2.test_prevalue;fold3.test_prevalue;fold4.test_prevalue;fold5.test_prevalue];
% ID = [fold1.test_ID;fold2.test_ID;fold3.test_ID;fold4.test_ID;fold5.test_ID];
% 手动版本结束，暂时不用手动版本-----------------------------------------
data = [ID,label,predict];
data_sort = sortrows(data,1);
%% best acc
pre = predict;
label_1 = label;
tmp_pre=sort(pre);
div_value=(tmp_pre(1:end-1)+tmp_pre(2:end))/2;
final_acc=zeros(length(div_value),1);
% for ii=1:length(div_value)
%    div_pre_label=double(pre>div_value(ii));
%    right=div_pre_label==label_1;
%    final_acc(ii)=sum(right)/length(label_1);
% end
for ii=1:length(div_value)
   div_pre_label=double(pre>div_value(ii));
   [ sensitivity, specificity] = sen2spc( label_1 ,div_pre_label);
   final_acc(ii)=sum([sensitivity,specificity]);
end

index=div_value(final_acc==max(final_acc));
figure;
plot(div_value,final_acc);title(['max is :',num2str(max(final_acc)),',thre is:',num2str(index(1))]);

%% auc
[X,Y,T,AUC] = perfcurve(abs(label_1-1),1-pre,1);
% [X,Y,T,AUC] = perfcurve(test_label,test_prevalue,1);
figure;
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification ')
legend(strcat('AUC:',num2str(AUC)),'Location','Best')


%% sen&spc
real_label = data_sort(:,2);
pre_label = data_sort(:,3)>index(1);


Test_orig_Label_LOO = double(real_label);
Test_pre_Label_LOO = double(pre_label);

%TP(True Positive)
TP = 0;
for i = 1:length(Test_orig_Label_LOO)
    if (Test_orig_Label_LOO(i,1) == 1) && (Test_pre_Label_LOO(i,1) == 1)
        TP = TP + 1;
    end
end

%TN(True Negative)
TN = 0;
for i = 1:length(Test_orig_Label_LOO)
    if (Test_orig_Label_LOO(i,1) == 0) && (Test_pre_Label_LOO(i,1) == 0)
        TN = TN + 1;
    end
end

%FP(False Positive)
FP = 0;
for i = 1:length(Test_orig_Label_LOO)
    if (Test_orig_Label_LOO(i,1) == 0) && (Test_pre_Label_LOO(i,1) == 1)
        FP = FP + 1;
    end
end
%FN(False Negative)
FN = 0;
for i = 1:length(Test_orig_Label_LOO)
    if (Test_orig_Label_LOO(i,1) == 1) && (Test_pre_Label_LOO(i,1) == 0)
        FN = FN + 1;
    end
end
%
A_accuracy = (TP + TN) / (TP + FN + FP + TN);
A_sensitivity = TP / (TP + FN);
A_specificity = TN / (TN + FP);

