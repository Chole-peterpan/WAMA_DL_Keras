function [aug_block,aug_block_othermode,augdict] = aug43D(block, aug_dict,othermode_flag,block_othermode)
% block 就是待扩增的3D块
% aug_block 扩增后的块，aug_block_othermode另一模态扩增的块，augdict该块扩增的具体参数

augdict = aug_dict;
% 先copy一份，如果不扩增的话就原样返回
aug_block = mat2gray(block);
if othermode_flag
    aug_block_othermode = mat2gray(block_othermode);
else
    aug_block_othermode = [];
end

% unifrnd (a,b)：生成均匀分布的随机数
% 可以利用此函数来构建：有多少几率下翻转这种情况
% 比如0.1几率，那么就生成0,1之间的数，如果小于等于0.1，就do，否则就不do

% 另外因为有双期同时输入的可能性存在，所以如果有双期，则生成一个随机数后，双期数据用同一参数扩增
%% 对比度调整
% 如果存在这个key，那么才操作，否则跳过
if isfield(augdict,'gray_adjust')
    if augdict.gray_adjust.flag
        % 如果没规定p，则默认为1
        if ~isfield(augdict.gray_adjust,'p')
            augdict.gray_adjust.p = 1;
        end
        
        % 生成均匀分布的随机数
        current_p = unifrnd (0,1);
        
        if current_p <= augdict.gray_adjust.p
            % 如果抽中奖，那么就进行扩增
            current_low = unifrnd(augdict.gray_adjust.low(1),augdict.gray_adjust.low(2));
            current_high = unifrnd(augdict.gray_adjust.up(1),augdict.gray_adjust.up(2));
            
            for i = 1:size(aug_block,3)
                aug_block(:,:,i) = imadjust(aug_block(:,:,i),[current_low,current_high],[0,1]);
            end
            % 如果有另外依摩泰的，则一起扩增
            if othermode_flag
                for i = 1:size(aug_block_othermode,3)
                    aug_block_othermode(:,:,i) = imadjust(aug_block_othermode(:,:,i),[current_low,current_high],[0,1]);
                end
            end
            
            % 记录扩增的具体参数到augdict里
            augdict.gray_adjust.current_low = current_low;
            augdict.gray_adjust.current_high = current_high;
            augdict.gray_adjust.do = true; %证明扩增成功了
        else
            % 如果没扩增，也要记录下来
            augdict.gray_adjust.do = false;
        end
    end
end
%% 左右反转
% qwe(:,end:-1:1)
if isfield(augdict,'LR_overturn')
    
    if augdict.LR_overturn.flag
        % 如果没规定p，则默认为1
        if ~isfield(augdict.LR_overturn,'p')
            augdict.LR_overturn.p = 1;
        end
        
        % 生成均匀分布的随机数
        current_p = unifrnd (0,1);
        
        
        if current_p <= augdict.LR_overturn.p
            % 如果抽中奖，那么就进行扩增
            for i = 1:size(aug_block,3)
                aug_block(:,:,i) = aug_block(:,end:-1:1,i);
            end
            % 如果有另外依摩泰的，则一起扩增
            if othermode_flag
                for i = 1:size(aug_block_othermode,3)
                    aug_block_othermode(:,:,i) = aug_block_othermode(:,end:-1:1,i);
                end
            end
            
            % 记录扩增的具体参数到augdict里
            augdict.LR_overturn.do = true;
            
        else
            % 如果没扩增，也要记录下来
            augdict.LR_overturn.do = false;
        end
    end
    
    
end



%% 上下翻转
% (:,end:-1:1)
if isfield(augdict,'UD_overturn')
    if augdict.UD_overturn.flag
        % 如果没规定p，则默认为1
        if ~isfield(augdict.UD_overturn,'p')
            augdict.UD_overturn.p = 1;
        end
        
        % 生成均匀分布的随机数
        current_p = unifrnd (0,1);
        
        
        if current_p <= augdict.UD_overturn.p
            % 如果抽中奖，那么就进行扩增
            for i = 1:size(aug_block,3)
                aug_block(:,:,i) = aug_block(end:-1:1 , : , i);
            end
            % 如果有另外依摩泰的，则一起扩增
            if othermode_flag
                for i = 1:size(aug_block_othermode,3)
                    aug_block_othermode(:,:,i) = aug_block_othermode(end:-1:1 , : , i);
                end
            end
            
            % 记录扩增的具体参数到augdict里
            augdict.UD_overturn.do = true;
            
        else
            % 如果没扩增，也要记录下来
            augdict.UD_overturn.do = false;
        end
    end
    
    
end



%% 输出mode ，这个是必有的
% mode 0，代表直接输出
% mode 1，代表 3Dresize
% mode 2，代表 容器居中 （注意，这个模式下目标dim3要和block的dim3相同）
% mode 3，代表 直接剪裁&填充 （当目标维度小于原始维度，则叫剪裁，当目标维度大于原始维度，则叫做填充padding）
%          不建议在dim3 padding，因为这样做没有意义（不过如果是做检测任务的话，可能就是有意义的） 
% mode 4，代表 居中剪裁&填充

% 如果未设置输出格式，则默认为mode 0
if ~isfield(augdict,'savefomat')
    augdict.savefomat.mode = 0;
end



if augdict.savefomat.mode == 0
    disp('or_size');%什么也不操作就完事儿了
    
    
elseif augdict.savefomat.mode == 1
    
    aug_block = imresize3(aug_block,  augdict.savefomat.param,  'cubic');
    if othermode_flag
        aug_block_othermode = imresize3(aug_block_othermode,  augdict.savefomat.param,  'cubic');
    end
    
    
    
elseif augdict.savefomat.mode == 2
    % （注意，这个模式下目标dim3要和block的dim3相同,且目标size的横截面必须是正方形）
    % 如果不是正方形，也可以正常运行，但是在逻辑上是错误的（逻辑错误出现在设置断点的那一句）
    if augdict.savefomat.param(3) ~= size(aug_block,3)
       error('dim3 between param & block shoud be same'); 
    end
    
    
    
    tmp_container = zeros(augdict.savefomat.param);
    if othermode_flag
        tmp_container_othermode = tmp_container;
    end
    
    
    param = augdict.savefomat.param;
    % 首先规定输出形状的横截面必须是正方形 ！
    % 如果x，y有任意一个维度是大于规定形状的，则必须先reshape,注意层数强制reshape为目标形状
    if max(size(aug_block,1),size(aug_block,2)) > param(1)
        % reshape:有两种情况（因为输入可能是立着的或倒着的长方形）的长方形或正方形，要保持原来的长宽比例来reshape
        if size(aug_block,1)>=size(aug_block,2)
            rate = size(aug_block,2)/size(aug_block,1);
            target_dim = floor(param(1) * rate);
            aug_block = imresize3(aug_block,[param(1),target_dim,param(3)],'cubic');%注意层数强制reshape为目标形状
            if othermode_flag
                aug_block_othermode = imresize3(aug_block_othermode,  [param(1),target_dim,param(3)],  'cubic');
            end
        else
            rate = size(aug_block,1)/size(aug_block,2);
            target_dim = floor(param(1) * rate);
            aug_block = imresize3(aug_block,[target_dim,param(2),param(3)],'cubic');%注意层数强制reshape为目标形状
            if othermode_flag
                aug_block_othermode = imresize3(aug_block_othermode, [target_dim,param(2),param(3)],  'cubic');
            end
        end
    end
    
    % ok，reshape完了，就要放到容器里了,一定要放在中间哦~
    d1_min = floor((param(1)-size(aug_block,1))/2);
    d2_min = floor((param(2)-size(aug_block,2))/2);
    
    tmp_container(d1_min+1:d1_min+size(aug_block,1) , d2_min+1:d2_min+size(aug_block,2), :) = aug_block;
    if othermode_flag
        tmp_container_othermode(d1_min+1:d1_min+size(aug_block,1) , d2_min+1:d2_min+size(aug_block,2), :) = aug_block_othermode;
    end
    
    % 重新把填充好的容器赋给block
    aug_block = tmp_container;
    if othermode_flag
        aug_block_othermode = tmp_container_othermode;
    end
    
% 直接剪裁，然后每个维度都从第一个开始放，具体对应到空间，matlab比较麻烦，暂时不对应    
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
    
    % 重新把填充好的容器赋给block
    aug_block = tmp_container;
    if othermode_flag
        aug_block_othermode = tmp_container_othermode;
    end
    
    
    
% 居中剪裁，其实就是 先居中剪裁  然后容器居中罢了
% 这个模式，dim3可以不相等
elseif augdict.savefomat.mode == 4
    param = augdict.savefomat.param;
    tmp_container = zeros(augdict.savefomat.param);
    if othermode_flag
        tmp_container_othermode = tmp_container;
    end
    
    % 先求出剪裁的尺寸  （注意，和直接剪裁不一样，要在中间剪裁）
    dim1_end = min(augdict.savefomat.param(1),size(aug_block,1));
    dim2_end = min(augdict.savefomat.param(2),size(aug_block,2));
    dim3_end = min(augdict.savefomat.param(3),size(aug_block,3));
    
    % 放入容器的最小index位置
    d1_min = floor((param(1)-dim1_end)/2);
    d2_min = floor((param(2)-dim2_end)/2);
    d3_min = floor((param(3)-dim3_end)/2);
    
    % 应该从block中间位置剪裁的方块对应的最小index位置（因为其实相当于，把block的中间部分抠出来，然后放到容器的中间）
    d1_min_4block = floor((size(aug_block,1)-dim1_end)/2);
    d2_min_4block = floor((size(aug_block,2)-dim2_end)/2);
    d3_min_4block = floor((size(aug_block,3)-dim3_end)/2);
    
    
    % 把block的中间部分抠出来，然后放到容器的中间
    tmp_container(d1_min+1:d1_min+dim1_end , d2_min+1:d2_min+dim2_end, d3_min+1:d3_min+dim3_end) = ...
        aug_block(d1_min_4block+1:d1_min_4block+dim1_end , d2_min_4block+1:d2_min_4block+dim2_end, d3_min_4block+1:d3_min_4block+dim3_end);
    if othermode_flag
        tmp_container_othermode(d1_min+1:d1_min+dim1_end , d2_min+1:d2_min+dim2_end, d3_min+1:d3_min+dim3_end) = ...
            aug_block_othermode(d1_min_4block+1:d1_min_4block+dim1_end , d2_min_4block+1:d2_min_4block+dim2_end, d3_min_4block+1:d3_min_4block+dim3_end);
    end
    
   
    % 重新把填充好的容器赋给block
    aug_block = tmp_container;
    if othermode_flag
        aug_block_othermode = tmp_container_othermode;
    end
    
    
    
    
    
    
    
else
    error('unknown savefomat');
    
    
end




%% 旋转：放在最后的原因是，返回mode可能是容器居中，先旋转会截断一部分，而先容器居中再旋转则不会
if isfield(augdict,'rotation')
    
    if augdict.rotation.flag
        % 如果没规定p，则默认为1
        if ~isfield(augdict.rotation,'p')
            augdict.rotation.p = 1;
        end
        
        % 生成均匀分布的随机数
        current_p = unifrnd (0,1);
        
        
        if current_p <= augdict.rotation.p
            % 如果抽中奖，那么就进行扩增
            angle = unifrnd(augdict.rotation.range(1),augdict.rotation.range(2));
            
            for i = 1:size(aug_block,3)
                aug_block(:,:,i) = imrotate(aug_block(:,:,i),angle,'bilinear','crop');
            end
            % 如果有另外依摩泰的，则一起扩增
            if othermode_flag
                for i = 1:size(aug_block_othermode,3)
                    aug_block_othermode(:,:,i) = imrotate(aug_block_othermode(:,:,i),angle,'bilinear','crop');
                end
            end
            
            % 记录扩增的具体参数到augdict里
            augdict.rotation.angle = angle;
            augdict.rotation.do = true; %证明扩增成功了
        else
            % 如果没扩增，也要记录下来
            augdict.rotation.do = false;
        end
    end
end


%% 将来加入其他的扩增方式，可以在format之前加入









end

