%% ����NIFTI������Ϣ�������浽mat�ļ���
%% ��ʼ��
initialization;
data_path = {};
data_window = {};
%% ����·������
% ģ̬or����1
data_path{end+1} = 'G:\estdata\data\0nii';
data_window{end+1} = [-25,285];
% ģ̬or����2
% data_path{end+1} = 'G:\estdata\data\0nii';
% data_window{end+1} = [-10,205];
% ģ̬or����3
% data_path{end+1} = 'G:\estdata\data\0nii';
% data_window{end+1} = [-400,805];
% label�ļ�
label_file = 'G:\estdata\data\label.xlsx';
% ����·��
save_path = 'G:\estdata\1mat';
%% �趨��������
% �Ƿ��ز�������size
resample_flag = true; 
% roi�����Ĵ�С��mm��
extend_length = 0; 
% �ռ������ز�����Ŀ��size
adjust_voxelsize = [0.5,0.5,1.0]; 
%% ִ��
STEP1_child;



