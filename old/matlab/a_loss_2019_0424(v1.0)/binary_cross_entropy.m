function [loss] = binary_cross_entropy(pre,label)
%
loss = abs(-1*label*log(pre)+(1-label)*log(1-pre));
end

