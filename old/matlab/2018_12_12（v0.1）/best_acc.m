

tmp_pre=sort(pre);
div_value=(tmp_pre(1:end-1)+tmp_pre(2:end))/2;
final_acc=zeros(length(div_value),1);
for ii=1:length(div_value)
   div_pre_label=double(pre>div_value(ii));
   right=div_pre_label==label;
   final_acc(ii)=sum(right)/length(label);
end
index=div_value(final_acc==max(final_acc));
plot(div_value,final_acc);title(['max is :',num2str(max(final_acc)),',thre is:',num2str(index(1))]);




[X1,Y1,T1,AUC1] = perfcurve(label1,pre1,1);%alexnet
[X2,Y2,T2,AUC2] = perfcurve(label2,pre2,1);%resnet43
[X3,Y3,T3,AUC3] = perfcurve(label3,pre3,1);%resnet20

figure;
% plot(X1,Y1,'r');hold on;
plot(X2,Y2,'g');hold on;
plot(X3,Y3,'b');
% r data   b  bis3  g  bis4
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification  ')
legend(strcat('AUC of ResNet24 :',num2str(AUC2)),strcat('AUC of ResNet51 :',num2str(AUC3)),'Location','Best')


