function [Data] = CT_data_preprocess(data,varargin)
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

    
%空间体素重采样（调整体素size）   
%输入的第二、三个参数，应该是原体素的x，y，z尺寸，以及新体素的x，y，z尺寸
%例子：[0.3,0.3,0.9]，[0.5,0.5,0.5]
% 1,2,3  0.5,0.5,0.5
elseif strcmp(varargin{1},'voxel_dim_resampling')
    % 获得原尺寸以及原voxelsize
    or_size = size(data);
    or_dim = varargin{2};
    % 获得目标voxelsize
    target_dim = varargin{3};
    % 计算各个dim的缩放比例
    dim1_rate = or_dim(1)/target_dim(1);
    dim2_rate = or_dim(2)/target_dim(2);
    dim3_rate = or_dim(3)/target_dim(3);
    % 求出最终resize的目标shape
    new_size = or_size.*[dim1_rate,dim2_rate,dim3_rate];
    % 3D reshape
    Data = imresize3(data,  new_size,  'cubic');
    
else
    error('no specified method');
end
end








