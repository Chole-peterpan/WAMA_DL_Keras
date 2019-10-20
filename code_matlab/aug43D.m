function [aug_block,augdict] = aug43D(block, aug_dict)
% block ���Ǵ�������3D��,��һ��cell
% aug_block ������Ŀ飬Ҳ��һ��cell
% aug_dict�����������ϸ�������õ�ԭ�ֵ�
% augdict���������������ϸ�������ֵ�


%% ��copyһ�ݣ�����������Ļ���ԭ�����أ�copy֮�����Թ�һ��
augdict = aug_dict;
aug_block = block;
for i = 1:length(aug_block)
    aug_block{i} = mat2gray(aug_block{i});
end

%%
% unifrnd (a,b)�����ɾ��ȷֲ��������
% �������ô˺������������ж��ټ����·�ת�������
% ����0.1���ʣ���ô������0,1֮����������С�ڵ���0.1����do������Ͳ�do

% ������Ϊ��˫��ͬʱ����Ŀ����Դ��ڣ����������˫�ڣ�������һ���������˫��������ͬһ��������
%% �������
% random_cut ����˵��
% flag���Ƿ���м���
% p ���ü��ĸ���
% dim��list�����Ҫ�ü���һά�ȣ��Ǿ͵���[1],����ü�ȫ��ά�ȣ������[1,2,3]
% range:list���ü���������ı仯��Χ������˵��ȥ���ķ�Χ������dim��Ӧ��dim�ü�����ά�ȣ���rangeҲӦ���ǳ���Ϊ3������
% Ԫ��ֵΪ0~1��0���ü���1ȫ�ü���ȥ����

% ���ӣ�augdict.random_cut.dim = [1,2,3]; augdict.random_cut.range = [0.1,0.2,0.3];
% ��ÿ��ά�ȶ��ü����ü��ķ�Χ��[0.1,0.2,0.3]

if isfield(augdict,'random_cut')
    if ~isfield(augdict.random_cut,'flag')
        augdict.random_cut.flag = 0;
        warning('no setting for cut flag, set flag false(or 0)');
    end
    
    if augdict.random_cut.flag
        % ���δ����p�����Զ���ȫΪ1
        if ~isfield(augdict.random_cut,'p')
            augdict.random_cut.p = 1;
        end
        % ���ɾ��ȷֲ��������
        current_p = unifrnd (0,1);
        if current_p <= augdict.random_cut.p
            % ������н�����ô�ͽ�������
            
            % ��ȡ����ü��ĳ���
            if ismember(1,augdict.random_cut.dim)
                dim1_cut = floor(size(aug_block{1},1) * unifrnd(0,augdict.random_cut.range(augdict.random_cut.dim == 1)));
            else
                dim1_cut = 0;
            end
            
            if ismember(2,augdict.random_cut.dim)
                dim2_cut = floor(size(aug_block{1},2) * unifrnd(0,augdict.random_cut.range(augdict.random_cut.dim == 2)));
            else
                dim2_cut = 0;
            end
            
            if ismember(3,augdict.random_cut.dim)
                dim3_cut = floor(size(aug_block{1},3) * unifrnd(0,augdict.random_cut.range(augdict.random_cut.dim == 3)));
            else
                dim3_cut = 0;
            end
            
            
            % ִ�вü�(��for��ÿ��ģ̬�Ķ�����)
            for i = 1:length(aug_block)
                aug_block{i} = aug_block{i}(1:end-dim1_cut, 1:end-dim2_cut, 1:end-dim3_cut);
            end
            
            % ����������ʵ�ʲ���
            augdict.random_cut.do = true;
            augdict.random_cut.dim1_cut = dim1_cut;
            augdict.random_cut.dim2_cut = dim2_cut;
            augdict.random_cut.dim3_cut = dim3_cut;
            
        else
            % ���û������ҲҪ��¼����
            augdict.random_cut.do = false;   
        end
    end
end





%% ������죺�����ο��漴���У�
% range �� ����ı���
% range_lowΪ������������ޣ����Ϊ0.5����ô����С��ԭ���ߴ��һ��
% range_highΪ�������������
% ����
% augdict.random_scale.dim = [1,2,3];
% augdict.random_scale.range_low = [1.1, 1.2, 1.3];
% augdict.random_scale.range_high = [1.1, 1.2, 1.3];

if isfield(augdict,'random_scale')
    if ~isfield(augdict.random_scale,'flag')
        augdict.random_scale.flag = 0;
        warning('no setting for scale flag, set flag false');
    end
    
    if augdict.random_scale.flag
        % ���δ����p�����Զ���ȫΪ1
        if ~isfield(augdict.random_scale,'p')
            augdict.random_scale.p = 1;
        end
        % ���ɾ��ȷֲ��������
        current_p = unifrnd (0,1);
        if current_p <= augdict.random_scale.p
            % ������н�����ô�ͽ�������
            
            % ��ȡ������ŵı���
            if ismember(1,augdict.random_scale.dim)
                index = augdict.random_scale.dim == 1;
                dim1_scale = unifrnd(augdict.random_scale.range_low(index),augdict.random_scale.range_high(index));
            else
                dim1_scale = 1;
            end
            
            if ismember(2,augdict.random_scale.dim)
                index = augdict.random_scale.dim == 2;
                dim2_scale = unifrnd(augdict.random_scale.range_low(index),augdict.random_scale.range_high(index));
            else
                dim2_scale = 1;
            end
            
            if ismember(3,augdict.random_scale.dim)
                index = augdict.random_scale.dim == 3;
                dim3_scale = unifrnd(augdict.random_scale.range_low(index),augdict.random_scale.range_high(index));
            else
                dim3_scale = 1;
            end
            
            % ִ������(ÿ��ģ̬)
            for i = 1:length(aug_block)
                aug_block{i} = imresize3(aug_block{i}, floor(size(aug_block{i}).*[dim1_scale, dim2_scale, dim3_scale]),  'cubic');
            end
            
            % ����������ʵ�ʲ���
            augdict.random_scale.do = true;
            augdict.random_scale.dim1_scale = dim1_scale;
            augdict.random_scale.dim2_scale = dim2_scale;
            augdict.random_scale.dim3_scale = dim3_scale;
            
        else
            % ���û������ҲҪ��¼����
            augdict.random_scale.do = false;  
        end
    end
end

%% Ť��

%% �漴��������-��������

%% �����������-��˹����

%% ����ڶ�
% ��ʲô���Ķ����ڼ�����

%% �Աȶȵ���
% ����������key����ô�Ų�������������
if isfield(augdict,'gray_adjust')
    if ~isfield(augdict.gray_adjust,'flag')
        augdict.gray_adjust.flag = 0;
        warning('no setting for gray_adjust flag, set flag false');
    end
    
    if augdict.gray_adjust.flag
        % ���û�涨p����Ĭ��Ϊ1
        if ~isfield(augdict.gray_adjust,'p')
            augdict.gray_adjust.p = 1;
        end
        
        % ���ɾ��ȷֲ��������
        current_p = unifrnd (0,1);
        
        if current_p <= augdict.gray_adjust.p
            % ������н�����ô�ͽ�������
            current_low = unifrnd(augdict.gray_adjust.low(1),augdict.gray_adjust.low(2));
            current_high = unifrnd(augdict.gray_adjust.up(1),augdict.gray_adjust.up(2));
            
            % ִ������
            for i = 1:length(aug_block)
                for ii = 1:size(aug_block{i},3)
                    aug_block{i}(:,:,ii) = imadjust(aug_block{i}(:,:,ii),[current_low,current_high],[0,1]);
                end
            end
            
            % ��¼�����ľ��������augdict��
            augdict.gray_adjust.current_low = current_low;
            augdict.gray_adjust.current_high = current_high;
            augdict.gray_adjust.do = true; %֤�������ɹ���
        else
            % ���û������ҲҪ��¼����
            augdict.gray_adjust.do = false;
        end
    end
end
%% ���ҷ�ת

if isfield(augdict,'LR_overturn')
    if ~isfield(augdict.LR_overturn,'flag')
        augdict.LR_overturn.flag = 0;
        warning('no setting for LR_overturn flag, set flag false');
    end
    
    if augdict.LR_overturn.flag
        % ���û�涨p����Ĭ��Ϊ1
        if ~isfield(augdict.LR_overturn,'p')
            augdict.LR_overturn.p = 1;
        end
        
        % ���ɾ��ȷֲ��������
        current_p = unifrnd (0,1);
        
        
        if current_p <= augdict.LR_overturn.p% ������н�����ô�ͽ�������
            
            % ִ������
            for i = 1:length(aug_block)
                for ii = 1:size(aug_block{i},3)
                    aug_block{i}(:,:,ii) = aug_block{i}(:,end:-1:1,ii);
                end
            end
            
            % ��¼�����ľ��������augdict��
            augdict.LR_overturn.do = true;
            
        else
            % ���û������ҲҪ��¼����
            augdict.LR_overturn.do = false;
        end
    end
end



%% ���·�ת

if isfield(augdict,'UD_overturn')
    if ~isfield(augdict.UD_overturn,'flag')
        augdict.UD_overturn.flag = 0;
        warning('no setting for UD_overturn flag, set flag false');
    end
    
    
    if augdict.UD_overturn.flag
        % ���û�涨p����Ĭ��Ϊ1
        if ~isfield(augdict.UD_overturn,'p')
            augdict.UD_overturn.p = 1;
        end
        
        % ���ɾ��ȷֲ��������
        current_p = unifrnd (0,1);
        
        
        if current_p <= augdict.UD_overturn.p% ������н�����ô�ͽ�������
            
            % ִ������
            for i = 1:length(aug_block)
                for ii = 1:size(aug_block{i},3)
                    aug_block{i}(:,:,ii) = aug_block{i}(end:-1:1 , : , ii);
                end
            end
            
            % ��¼�����ľ��������augdict��
            augdict.UD_overturn.do = true;
            
        else
            % ���û������ҲҪ��¼����
            augdict.UD_overturn.do = false;
        end
    end
end



%% ���mode ������Ǳ��е�,���Բ���Ҫflag
% mode 0��ֱ�����
% mode 1��ǿ��3Dresize�����size
% mode 2���������� ��ע�⣬���ģʽ��Ŀ��dim3Ҫ��block��dim3��ͬ��,�����״������������reshape����Ӧ����
% mode 3��ֱ�Ӽ���&��� ����Ŀ��ά��С��ԭʼά�ȣ���м��ã���Ŀ��ά�ȴ���ԭʼά�ȣ���������padding��
%          ��������dim3 padding����Ϊ������û�����壨������������������Ļ������ܾ���������ģ�
% mode 4�����м���&���

% ���δ���������ʽ����Ĭ��Ϊmode 0
if ~isfield(augdict,'savefomat')
    augdict.savefomat.mode = 0;
end


% mode 0 ==================================================================
if augdict.savefomat.mode == 0
    disp('or_size');%ʲôҲ�����������¶���
    
% mode 1 ==================================================================    
elseif augdict.savefomat.mode == 1
    
    for i = 1:length(aug_block)
        aug_block{i} = imresize3(aug_block{i},  augdict.savefomat.param,  'cubic');
    end
    
    
% mode 2 ==================================================================    
elseif augdict.savefomat.mode == 2
    % ��ע�⣬���ģʽ��Ŀ��dim3Ҫ��block��dim3��ͬ,��Ŀ��size�ĺ��������������Σ�
    % ������������Σ�Ҳ�����������У��������߼����Ǵ���ģ��߼�������������öϵ����һ�䣩
    if augdict.savefomat.param(3) ~= size(aug_block{1},3)
        error('dim3 between param & block shoud be same');
    end
      
    tmp_container = {};
    for i = 1:length(aug_block)
       tmp_container{i} =  zeros(augdict.savefomat.param);
    end

    param = augdict.savefomat.param;
    % ���ȹ涨�����״�ĺ��������������� ��
    % ���x��y������һ��ά���Ǵ��ڹ涨��״�ģ��������reshape,ע�����ǿ��reshapeΪĿ����״
    if max(size(aug_block{1},1),size(aug_block{1},2)) > param(1)
        % reshape:�������������Ϊ������������ŵĻ��ŵĳ����Σ��ĳ����λ������Σ�Ҫ����ԭ���ĳ��������reshape
        if size(aug_block{1},1)>=size(aug_block{1},2)
            rate = size(aug_block{1},2)/size(aug_block{1},1);
            target_dim = floor(param(1) * rate);
            % ִ��reshape
            for i = 1:length(aug_block)
                aug_block{i} = imresize3(aug_block{i},[param(1),target_dim,param(3)],'cubic');%ע�����ǿ��reshapeΪĿ����״
            end
            
        else
            rate = size(aug_block{1},1)/size(aug_block{1},2);
            target_dim = floor(param(1) * rate);
            % ִ��reshape
            for i = 1:length(aug_block)
                aug_block{i} = imresize3(aug_block{i},[target_dim,param(2),param(3)],'cubic');%ע�����ǿ��reshapeΪĿ����״
            end
        end
    end
    
    % ok��reshape���ˣ���Ҫ�ŵ���������,һ��Ҫ�����м�Ŷ~
    % �ȼ����·ŵ�λ�ã�dim index�½磩
    d1_min = floor((param(1)-size(aug_block{1},1))/2);
    d2_min = floor((param(2)-size(aug_block{1},2))/2);
    
    % ִ�з���
    for i = 1:length(aug_block)
        tmp_container{i}(d1_min+1:d1_min+size(aug_block{1},1) , d2_min+1:d2_min+size(aug_block{1},2), :) = aug_block{i};
    end
    
    % ���°����õ���������block
    aug_block = tmp_container;


    
% mode 3 ==================================================================    
% ֱ�ӿ����ϽǼ���or padding������棩��Ȼ��ÿ��ά�ȶ��ӵ�һ����ʼ�ţ������Ӧ���ռ������ᣬmatlab�Ƚ��鷳����ʱ����Ӧ��
elseif augdict.savefomat.mode == 3
    
    tmp_container = {};
    for i = 1:length(aug_block)
       tmp_container{i} =  zeros(augdict.savefomat.param);
    end
    
    tmp_size = size(tmp_container{1});
    or_size  = size(aug_block{1});
    
    dim1_end = min(tmp_size(1),or_size(1));
    dim2_end = min(tmp_size(2),or_size(2));
    dim3_end = min(tmp_size(3),or_size(3));
    
    % ִ�м���
    for i = 1:length(aug_block)
        tmp_container{i}(1:dim1_end,  1:dim2_end,  1:dim3_end) = aug_block{i}(1:dim1_end,  1:dim2_end,  1:dim3_end);
    end
    
    % ���°����õ���������block
    aug_block = tmp_container;
    
    
% mode 4 ==================================================================
% ���м��ã���ʵ���� �Ⱦ��м���  Ȼ���������а���
% ���ģʽ��dim3���Բ����
elseif augdict.savefomat.mode == 4
    param = augdict.savefomat.param;

    tmp_container = {};
    for i = 1:length(aug_block)
       tmp_container{i} =  zeros(augdict.savefomat.param);
    end
    
    % ��������õĳߴ�  ��ע�⣬��ֱ�Ӽ��ò�һ����Ҫ���м���ã�
    dim1_end = min(augdict.savefomat.param(1),size(aug_block{1},1));
    dim2_end = min(augdict.savefomat.param(2),size(aug_block{1},2));
    dim3_end = min(augdict.savefomat.param(3),size(aug_block{1},3));
    
    % ������������Сindexλ��
    d1_min = floor((param(1)-dim1_end)/2);
    d2_min = floor((param(2)-dim2_end)/2);
    d3_min = floor((param(3)-dim3_end)/2);
    
    % Ӧ�ô�block�м�λ�ü��õķ����Ӧ����Сindexλ�ã���Ϊ��ʵ�൱�ڣ���block���м䲿�ֿٳ�����Ȼ��ŵ��������м䣩
    d1_min_4block = floor((size(aug_block{1},1)-dim1_end)/2);
    d2_min_4block = floor((size(aug_block{1},2)-dim2_end)/2);
    d3_min_4block = floor((size(aug_block{1},3)-dim3_end)/2);
    
    
    % ��block���м䲿�ֿٳ�����Ȼ��ŵ��������м�
    for i =1:length(aug_block)
        tmp_container{i}(d1_min+1:d1_min+dim1_end , d2_min+1:d2_min+dim2_end, d3_min+1:d3_min+dim3_end) = ...
            aug_block{i}(d1_min_4block+1:d1_min_4block+dim1_end , d2_min_4block+1:d2_min_4block+dim2_end, d3_min_4block+1:d3_min_4block+dim3_end);
    end
    
    
    % ���°����õ���������block
    aug_block = tmp_container;
    
else
    error('unknown savefomat');
    
    
end


%% ��ת����������ԭ���ǣ�����mode�������������У�����ת��ض�һ���֣�����������������ת�򲻻�
if isfield(augdict,'rotation')
    if ~isfield(augdict.rotation,'flag')
        augdict.rotation.flag = 0;
        warning('no setting for rotation flag, set flag false');
    end
    
    
    if augdict.rotation.flag
        % ���û�涨p����Ĭ��Ϊ1
        if ~isfield(augdict.rotation,'p')
            augdict.rotation.p = 1;
        end
        
        % ���ɾ��ȷֲ��������
        current_p = unifrnd (0,1);
        
        
        if current_p <= augdict.rotation.p % ������н�����ô�ͽ�������
            % ����������ĽǶ�
            angle = unifrnd(augdict.rotation.range(1),augdict.rotation.range(2));
            % ִ������
            for i = 1:length(aug_block)
                for ii = 1:size(aug_block{1},3)
                    aug_block{i}(:,:,ii) = imrotate(aug_block{i}(:,:,ii),angle,'bilinear','crop');
                end
            end
            % ��¼�����ľ��������augdict��
            augdict.rotation.angle = angle;
            augdict.rotation.do = true; %֤�������ɹ���
        else
            % ���û������ҲҪ��¼����
            augdict.rotation.do = false;
        end
    end
end


%% ��������������������ʽ��������format֮ǰ����









end

