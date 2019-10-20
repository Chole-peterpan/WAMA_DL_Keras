function [Data] = tumor_preprocess(data)
%Ԥ������
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

% normolization(�Ѿ���׼����������Ҫ��һ������Ϊ�ں�������Աȶȵ�ʱ������Ĭ����0��1֮��ģ��������һ�������ضϴ󲿷�ֵ����ͼƬʧ�棩
% ������������ʱ���Ѿ��ڵ����Աȶ�ǰ���й�һ���ˣ�������һ������ν������ȥҲ����
minn = min(min(min(data_nr)));
maxx = max(max(max(data_nr)));
data_no = (data_nr - minn)/(maxx - minn);

% normolization (method 2)====================
%     data_no1 = mat2gray(data_nr);

Data = data_no;
end

