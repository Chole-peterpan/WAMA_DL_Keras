%% �۲����ս����watcher����realtimewatcher��һ�����������������۲��
% ��Ϊ3������1��ȫ��&��Ե��������ˣ�2����Ե��������˵ĸ���������3����Ե�ĳ����������block

%% ��ʼ��
close all;
clear;
clc;

%% ָ������
logpath = 'H:\qwe\log';
foldname = 1;
test_flag = true;
acc_thresold = 0.5;% ׼ȷ����ֵ

%% ��������
result_watcher_1

%% ��һ������ʾ������Ϊ��λ��loss��pre��acc���Կ���ƽ����
% �趨����
scale_size = 300;%loss����ͼ���ŵı���
% ��ʾloss�Լ�acc
result_watcher_2;



%% �ڶ�������ʾ����һ���趨����
% ��ʾ����id
uniq_id = unique(block_name(:,1))';
disp(['id:',num2str(uniq_id)]);
disp(person_id);


scale_size = 300;%loss����ͼ���ŵı���
% �趨����
s_id = 51;%ָ��������ţ���������0����ʾ
v_id = 1;%ָ��ɨ�����ţ���������0����ʾ
m_id = 1;%ָ��������ţ���������0����ʾ
%��Ϊ����n������ʾidΪn�Ĳ��˵����п�������acc���Լ�loss������ʾloss�����ַ�����һ���Ƕ������ߣ�һ����heatmap��heatmap��Ҫ�Լ��任ΪͼƬ��ʽ��ʾ
%��ʾ��ʱ��ע�����򣬼�ͬһ�����Ĳ�ͬ����ð���ԭ���Ŀռ����λ��������ʾ����������ͼ������ͺ�������Ƭ������һ��

show_flag = true;%���Ϊtrue������ʾ����ָ��
result_watcher_3;






















