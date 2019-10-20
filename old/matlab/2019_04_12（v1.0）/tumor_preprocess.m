function [Data] = tumor_preprocess(data)
%预处理函数
disp('preprocessing...');

[xx,yy,zz]=size(data);

% adjust CT window
data(data<-30)=-30;
data(data>280)=280;

% standardization
data_flatten=reshape(data,[1,xx*yy*zz]);
meann = mean(data_flatten); % do not code like this :meann = mean(mean(mean(data)));
stdd = std(data_flatten);  % do not code like this :stdd = std(std(std(data)));
data_nr = (data-meann)/stdd;

% standardization(method 2)===================
%     data_nr1 = zscore(data_flatten);
%     data_nr1 = reshape(data_nr1,size(data));

% normolization(已经标准化，但仍需要归一化，因为在后面调整对比度的时候输入默认是0到1之间的，如果不归一化，则会截断大部分值导致图片失真）
% 不过在扩增的时候，已经在调整对比度前进行归一化了，所以这一步无所谓，加上去也可以
minn = min(min(min(data_nr)));
maxx = max(max(max(data_nr)));
data_no = (data_nr - minn)/(maxx - minn);

% normolization (method 2)====================
%     data_no1 = mat2gray(data_nr);

Data = data_no;
end

