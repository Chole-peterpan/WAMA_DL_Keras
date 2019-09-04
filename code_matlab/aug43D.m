function [aug_block,aug_block_othermode,augdict] = aug43D(block, aug_dict,othermode_flag,block_othermode)
% block ���Ǵ�������3D��
% aug_block ������Ŀ飬aug_block_othermode��һģ̬�����Ŀ飬augdict�ÿ������ľ������

augdict = aug_dict;
% ��copyһ�ݣ�����������Ļ���ԭ������
aug_block = mat2gray(block);
if othermode_flag
    aug_block_othermode = mat2gray(block_othermode);
else
    aug_block_othermode = [];
end

% unifrnd (a,b)�����ɾ��ȷֲ��������
% �������ô˺������������ж��ټ����·�ת�������
% ����0.1���ʣ���ô������0,1֮����������С�ڵ���0.1����do������Ͳ�do

% ������Ϊ��˫��ͬʱ����Ŀ����Դ��ڣ����������˫�ڣ�������һ���������˫��������ͬһ��������
%% �Աȶȵ���
% ����������key����ô�Ų�������������
if isfield(augdict,'gray_adjust')
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
            
            for i = 1:size(aug_block,3)
                aug_block(:,:,i) = imadjust(aug_block(:,:,i),[current_low,current_high],[0,1]);
            end
            % �����������Ħ̩�ģ���һ������
            if othermode_flag
                for i = 1:size(aug_block_othermode,3)
                    aug_block_othermode(:,:,i) = imadjust(aug_block_othermode(:,:,i),[current_low,current_high],[0,1]);
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
% qwe(:,end:-1:1)
if isfield(augdict,'LR_overturn')
    
    if augdict.LR_overturn.flag
        % ���û�涨p����Ĭ��Ϊ1
        if ~isfield(augdict.LR_overturn,'p')
            augdict.LR_overturn.p = 1;
        end
        
        % ���ɾ��ȷֲ��������
        current_p = unifrnd (0,1);
        
        
        if current_p <= augdict.LR_overturn.p
            % ������н�����ô�ͽ�������
            for i = 1:size(aug_block,3)
                aug_block(:,:,i) = aug_block(:,end:-1:1,i);
            end
            % �����������Ħ̩�ģ���һ������
            if othermode_flag
                for i = 1:size(aug_block_othermode,3)
                    aug_block_othermode(:,:,i) = aug_block_othermode(:,end:-1:1,i);
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
% (:,end:-1:1)
if isfield(augdict,'UD_overturn')
    if augdict.UD_overturn.flag
        % ���û�涨p����Ĭ��Ϊ1
        if ~isfield(augdict.UD_overturn,'p')
            augdict.UD_overturn.p = 1;
        end
        
        % ���ɾ��ȷֲ��������
        current_p = unifrnd (0,1);
        
        
        if current_p <= augdict.UD_overturn.p
            % ������н�����ô�ͽ�������
            for i = 1:size(aug_block,3)
                aug_block(:,:,i) = aug_block(end:-1:1 , : , i);
            end
            % �����������Ħ̩�ģ���һ������
            if othermode_flag
                for i = 1:size(aug_block_othermode,3)
                    aug_block_othermode(:,:,i) = aug_block_othermode(end:-1:1 , : , i);
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



%% ���mode ������Ǳ��е�
% mode 0������ֱ�����
% mode 1������ 3Dresize
% mode 2������ �������� ��ע�⣬���ģʽ��Ŀ��dim3Ҫ��block��dim3��ͬ��
% mode 3������ ֱ�Ӽ���&��� ����Ŀ��ά��С��ԭʼά�ȣ���м��ã���Ŀ��ά�ȴ���ԭʼά�ȣ���������padding��
%          ��������dim3 padding����Ϊ������û�����壨������������������Ļ������ܾ���������ģ� 
% mode 4������ ���м���&���

% ���δ���������ʽ����Ĭ��Ϊmode 0
if ~isfield(augdict,'savefomat')
    augdict.savefomat.mode = 0;
end



if augdict.savefomat.mode == 0
    disp('or_size');%ʲôҲ�����������¶���
    
    
elseif augdict.savefomat.mode == 1
    
    aug_block = imresize3(aug_block,  augdict.savefomat.param,  'cubic');
    if othermode_flag
        aug_block_othermode = imresize3(aug_block_othermode,  augdict.savefomat.param,  'cubic');
    end
    
    
    
elseif augdict.savefomat.mode == 2
    % ��ע�⣬���ģʽ��Ŀ��dim3Ҫ��block��dim3��ͬ,��Ŀ��size�ĺ��������������Σ�
    % ������������Σ�Ҳ�����������У��������߼����Ǵ���ģ��߼�������������öϵ����һ�䣩
    if augdict.savefomat.param(3) ~= size(aug_block,3)
       error('dim3 between param & block shoud be same'); 
    end
    
    
    
    tmp_container = zeros(augdict.savefomat.param);
    if othermode_flag
        tmp_container_othermode = tmp_container;
    end
    
    
    param = augdict.savefomat.param;
    % ���ȹ涨�����״�ĺ��������������� ��
    % ���x��y������һ��ά���Ǵ��ڹ涨��״�ģ��������reshape,ע�����ǿ��reshapeΪĿ����״
    if max(size(aug_block,1),size(aug_block,2)) > param(1)
        % reshape:�������������Ϊ������������ŵĻ��ŵĳ����Σ��ĳ����λ������Σ�Ҫ����ԭ���ĳ��������reshape
        if size(aug_block,1)>=size(aug_block,2)
            rate = size(aug_block,2)/size(aug_block,1);
            target_dim = floor(param(1) * rate);
            aug_block = imresize3(aug_block,[param(1),target_dim,param(3)],'cubic');%ע�����ǿ��reshapeΪĿ����״
            if othermode_flag
                aug_block_othermode = imresize3(aug_block_othermode,  [param(1),target_dim,param(3)],  'cubic');
            end
        else
            rate = size(aug_block,1)/size(aug_block,2);
            target_dim = floor(param(1) * rate);
            aug_block = imresize3(aug_block,[target_dim,param(2),param(3)],'cubic');%ע�����ǿ��reshapeΪĿ����״
            if othermode_flag
                aug_block_othermode = imresize3(aug_block_othermode, [target_dim,param(2),param(3)],  'cubic');
            end
        end
    end
    
    % ok��reshape���ˣ���Ҫ�ŵ���������,һ��Ҫ�����м�Ŷ~
    d1_min = floor((param(1)-size(aug_block,1))/2);
    d2_min = floor((param(2)-size(aug_block,2))/2);
    
    tmp_container(d1_min+1:d1_min+size(aug_block,1) , d2_min+1:d2_min+size(aug_block,2), :) = aug_block;
    if othermode_flag
        tmp_container_othermode(d1_min+1:d1_min+size(aug_block,1) , d2_min+1:d2_min+size(aug_block,2), :) = aug_block_othermode;
    end
    
    % ���°����õ���������block
    aug_block = tmp_container;
    if othermode_flag
        aug_block_othermode = tmp_container_othermode;
    end
    
% ֱ�Ӽ��ã�Ȼ��ÿ��ά�ȶ��ӵ�һ����ʼ�ţ������Ӧ���ռ䣬matlab�Ƚ��鷳����ʱ����Ӧ    
elseif augdict.savefomat.mode == 3
    tmp_container = zeros(augdict.savefomat.param);
    if othermode_flag
        tmp_container_othermode = tmp_container;
    end
    
    tmp_size = size(tmp_container);
    or_size  = size(aug_block);
    
    dim1_end = min(tmp_size(1),or_size(1));
    dim2_end = min(tmp_size(2),or_size(2));
    dim3_end = min(tmp_size(3),or_size(3));
    
    tmp_container(1:dim1_end,  1:dim2_end,  1:dim3_end) = aug_block(1:dim1_end,  1:dim2_end,  1:dim3_end);
    if othermode_flag
        tmp_container_othermode(1:dim1_end,  1:dim2_end,  1:dim3_end) = aug_block_othermode(1:dim1_end,  1:dim2_end,  1:dim3_end);
    end
    
    % ���°����õ���������block
    aug_block = tmp_container;
    if othermode_flag
        aug_block_othermode = tmp_container_othermode;
    end
    
    
    
% ���м��ã���ʵ���� �Ⱦ��м���  Ȼ���������а���
% ���ģʽ��dim3���Բ����
elseif augdict.savefomat.mode == 4
    param = augdict.savefomat.param;
    tmp_container = zeros(augdict.savefomat.param);
    if othermode_flag
        tmp_container_othermode = tmp_container;
    end
    
    % ��������õĳߴ�  ��ע�⣬��ֱ�Ӽ��ò�һ����Ҫ���м���ã�
    dim1_end = min(augdict.savefomat.param(1),size(aug_block,1));
    dim2_end = min(augdict.savefomat.param(2),size(aug_block,2));
    dim3_end = min(augdict.savefomat.param(3),size(aug_block,3));
    
    % ������������Сindexλ��
    d1_min = floor((param(1)-dim1_end)/2);
    d2_min = floor((param(2)-dim2_end)/2);
    d3_min = floor((param(3)-dim3_end)/2);
    
    % Ӧ�ô�block�м�λ�ü��õķ����Ӧ����Сindexλ�ã���Ϊ��ʵ�൱�ڣ���block���м䲿�ֿٳ�����Ȼ��ŵ��������м䣩
    d1_min_4block = floor((size(aug_block,1)-dim1_end)/2);
    d2_min_4block = floor((size(aug_block,2)-dim2_end)/2);
    d3_min_4block = floor((size(aug_block,3)-dim3_end)/2);
    
    
    % ��block���м䲿�ֿٳ�����Ȼ��ŵ��������м�
    tmp_container(d1_min+1:d1_min+dim1_end , d2_min+1:d2_min+dim2_end, d3_min+1:d3_min+dim3_end) = ...
        aug_block(d1_min_4block+1:d1_min_4block+dim1_end , d2_min_4block+1:d2_min_4block+dim2_end, d3_min_4block+1:d3_min_4block+dim3_end);
    if othermode_flag
        tmp_container_othermode(d1_min+1:d1_min+dim1_end , d2_min+1:d2_min+dim2_end, d3_min+1:d3_min+dim3_end) = ...
            aug_block_othermode(d1_min_4block+1:d1_min_4block+dim1_end , d2_min_4block+1:d2_min_4block+dim2_end, d3_min_4block+1:d3_min_4block+dim3_end);
    end
    
   
    % ���°����õ���������block
    aug_block = tmp_container;
    if othermode_flag
        aug_block_othermode = tmp_container_othermode;
    end
    
    
    
    
    
    
    
else
    error('unknown savefomat');
    
    
end




%% ��ת����������ԭ���ǣ�����mode�������������У�����ת��ض�һ���֣�����������������ת�򲻻�
if isfield(augdict,'rotation')
    
    if augdict.rotation.flag
        % ���û�涨p����Ĭ��Ϊ1
        if ~isfield(augdict.rotation,'p')
            augdict.rotation.p = 1;
        end
        
        % ���ɾ��ȷֲ��������
        current_p = unifrnd (0,1);
        
        
        if current_p <= augdict.rotation.p
            % ������н�����ô�ͽ�������
            angle = unifrnd(augdict.rotation.range(1),augdict.rotation.range(2));
            
            for i = 1:size(aug_block,3)
                aug_block(:,:,i) = imrotate(aug_block(:,:,i),angle,'bilinear','crop');
            end
            % �����������Ħ̩�ģ���һ������
            if othermode_flag
                for i = 1:size(aug_block_othermode,3)
                    aug_block_othermode(:,:,i) = imrotate(aug_block_othermode(:,:,i),angle,'bilinear','crop');
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

