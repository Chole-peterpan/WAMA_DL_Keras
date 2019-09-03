function [Data] = tumor_preprocess(data,varargin)
%预处理函数
disp('preprocessing...');


%调整窗宽窗位
if strcmp(varargin{1},'window_change')
    
    disp('window_change');
    if varargin{2}(1) > varargin{2}(2)
        warning('CT_low can not be bigger than CT_high, switch by force')
        tmp = varargin{2}(1);
        varargin{2}(1)=varargin{2}(2);
        varargin{2}(2)=tmp;
    end
    data(data<varargin{2}(1))=varargin{2}(1);
    data(data>varargin{2}(2))=varargin{2}(2);
    Data = data;
    
%标准化    
elseif strcmp(varargin{1},'standardization')
    disp('standardization');
    [xx,yy,zz]=size(data);
    data_flatten=reshape(data,[1,xx*yy*zz]);
    meann = mean(data_flatten); % do not code like this :meann = mean(mean(mean(data)));
    stdd = std(data_flatten);  % do not code like this :stdd = std(std(std(data)));
    Data = (data-meann)/stdd;
    % standardization(method 2)===================
    %     data_nr1 = zscore(data_flatten);
    %     data_nr1 = reshape(data_nr1,size(data));    
    
%线性归一化
elseif strcmp(varargin{1},'Linear_normalization')
    disp('Linear_normalization');
    % normolization(已经标准化，但仍需要归一化，因为在后面调整对比度的时候输入默认是0到1之间的，如果不归一化，则会截断大部分值导致图片失真）
    % 不过在扩增的时候，已经在调整对比度前进行归一化了，所以这一步无所谓，加上去也可以
    minn = min(min(min(data)));
    maxx = max(max(max(data)));
    Data = (data - minn)/(maxx - minn);
    
    % normolization (method 2)====================
    % data_no1 = mat2gray(data_nr);

else
    error('no specified method');
end
end

