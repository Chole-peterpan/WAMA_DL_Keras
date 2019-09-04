function [Data] = CT_data_preprocess(data,varargin)
%Ԥ������
disp('preprocessing...');


%��������λ
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
    
%��׼��    
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
    
%���Թ�һ��
elseif strcmp(varargin{1},'Linear_normalization')
    disp('Linear_normalization');
    % normolization(�Ѿ���׼����������Ҫ��һ������Ϊ�ں�������Աȶȵ�ʱ������Ĭ����0��1֮��ģ��������һ�������ضϴ󲿷�ֵ����ͼƬʧ�棩
    % ������������ʱ���Ѿ��ڵ����Աȶ�ǰ���й�һ���ˣ�������һ������ν������ȥҲ����
    minn = min(min(min(data)));
    maxx = max(max(max(data)));
    Data = (data - minn)/(maxx - minn);
    
    % normolization (method 2)====================
    % data_no1 = mat2gray(data_nr);

    
%�ռ������ز�������������size��   
%����ĵڶ�������������Ӧ����ԭ���ص�x��y��z�ߴ磬�Լ������ص�x��y��z�ߴ�
%���ӣ�[0.3,0.3,0.9]��[0.5,0.5,0.5]
% 1,2,3  0.5,0.5,0.5
elseif strcmp(varargin{1},'voxel_dim_resampling')
    % ���ԭ�ߴ��Լ�ԭvoxelsize
    or_size = size(data);
    or_dim = varargin{2};
    % ���Ŀ��voxelsize
    target_dim = varargin{3};
    % �������dim�����ű���
    dim1_rate = or_dim(1)/target_dim(1);
    dim2_rate = or_dim(2)/target_dim(2);
    dim3_rate = or_dim(3)/target_dim(3);
    % �������resize��Ŀ��shape
    new_size = or_size.*[dim1_rate,dim2_rate,dim3_rate];
    % 3D reshape
    Data = imresize3(data,  new_size,  'cubic');
    
else
    error('no specified method');
end
end








