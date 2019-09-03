function [Data] = tumor_preprocess(data,varargin)
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

else
    error('no specified method');
end
end

