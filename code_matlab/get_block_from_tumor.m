function [blocks_all,mask_all,blocks] = get_block_from_tumor(data,mask,step,deepth)

blocks_all = {};
mask_all = {};
%deepth为0时，原封不动的返回即可
if deepth == 0
    blocks_all{1}=data;
    mask_all{1}=mask;
    blocks = 1;
else
    if size(mask,3)<deepth
        tmp_data = zeros(size(mask,1),size(mask,2),deepth);
        tmp_mask = tmp_data;
        tmp_data(:,:,1:size(mask,3)) = data;
        tmp_mask(:,:,1:size(mask,3)) = mask;
        Data = tmp_data;%最终block
        Mask = tmp_mask;%最终block mask
        
        blocks_all{1}=Data;
        mask_all{1}=Mask;
        blocks = 1;
    else
        blocks = floor((size(mask,3)-deepth)/step)+2;
        % 如果滑动到最后正好能取完全部层，则-1，因为这个代码无论如何都要从后往前滑动取一个block
        if mod((size(mask,3)-deepth),step)==0
            blocks=blocks-1;
        end
        % 开始滑动
        for iii = 1:(blocks-1)
            Data = zeros(size(mask,1),size(mask,2),deepth);
            Mask = Data;
            Data = data(:,:,(1+(iii-1)*step):(deepth+(iii-1)*step));
            Mask = mask(:,:,(1+(iii-1)*step):(deepth+(iii-1)*step));
            blocks_all{end+1}=Data;
            mask_all{end+1}=Mask;
        end
        
        % 从后往前滑动取一个block
        Data = zeros(size(mask,1),size(mask,2),deepth);
        Mask = Data;
        Data = data(:,:,end-(deepth-1):end);
        Mask = mask(:,:,end-(deepth-1):end);
        blocks_all{end+1}=Data;
        mask_all{end+1}=Mask;
    end
end

end





















