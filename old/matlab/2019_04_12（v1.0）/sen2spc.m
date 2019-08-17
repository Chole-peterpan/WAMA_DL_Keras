function [ A_sensitivity, A_specificity] = sen2spc( real_label ,pre_label)


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

A_sensitivity = TP / (TP + FN);
A_specificity = TN / (TN + FP);

end

