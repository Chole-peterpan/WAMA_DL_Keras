%% �����������֤���������������ƽ�⣬�����и���������ƽ��
%% ��ʼ��
initialization;

%% ����·��
block_mat_path =  'G:\estdata\2block';
augdict.mat_savepath =    'G:\estdata\4aug';

%% ����������������

%�趨���������ģʽ�Լ�size
augdict.savefomat.mode = 3;
augdict.savefomat.param = [150,150,80];

%����������
augdict.aug_num = 1000;
%�Եڼ���label�ķ���������ƽ��
augdict.balance_label_index = 2;
%ƽ��ʱ�����ı�����labelֵ��С����
augdict.a_b_ratio = [1,1];%������������a��b������AB����Ϊ1:2��������Ϊ[1,2],A�Ƿ�PD��B��PD

% ��������ת
augdict.rotation.flag = true;%�Ƿ�ʹ����ת����
augdict.rotation.range = [0,180];%��ת�����ĽǶȷ�Χ
augdict.rotation.p = 1.0;

% �������Աȶȵ���
augdict.gray_adjust.flag = true;%�Ƿ�ʹ�öԱȶȵ���
augdict.gray_adjust.up = [0.9,1];%�Աȶȵ������Ͻ緶Χ
augdict.gray_adjust.low = [.0,0.1];%�Աȶȵ������½緶Χ
augdict.gray_adjust.p = 1.0;

% ���������ҷ�ת LEFT RIGHT
augdict.LR_overturn.flag = true;%�Ƿ�ʹ�����ҷ�ת
augdict.LR_overturn.p = 1.0;%���ҷ�ת�ĸ���

% ���������·�ת UP DOWN
augdict.UD_overturn.flag = true;%�Ƿ�ʹ�����·�ת
augdict.UD_overturn.p = 1.0;%���·�ת�ĸ���

% �������������
augdict.random_cut.flag = true;%�Ƿ���м���
augdict.random_cut.p = 1.0;
augdict.random_cut.dim = [1,2,3]; 
augdict.random_cut.range = [0.1,0.2,0.3];

% �������������
augdict.random_scale.flag = true;%�Ƿ���м���
augdict.random_scale.p = 1.0;
augdict.random_scale.dim = [1,2,3];
augdict.random_scale.range_low = [0.5, 0.6, 0.7];
augdict.random_scale.range_high = [1.1, 1.2, 1.3];

%% run
STEP4_child;


