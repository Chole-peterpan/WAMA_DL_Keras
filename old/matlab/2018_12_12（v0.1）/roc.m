

[X,Y,T,AUC] = perfcurve(label,pre,1);

plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Logistic Regression')
legend(strcat('AUC:',num2str(AUC)),'Location','Best')



