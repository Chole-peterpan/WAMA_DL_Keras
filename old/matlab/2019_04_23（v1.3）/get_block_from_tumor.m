function [blocks_all,mask_all,blocks] = get_block_from_tumor(data,mask)


blocks_all = {};
mask_all = {};

if size(mask,3)<16
    tmp_data = zeros(size(mask,1),size(mask,2),16);
    tmp_mask = tmp_data;
    tmp_data(:,:,1:size(mask,3)) = data;
    tmp_mask(:,:,1:size(mask,3)) = mask;
    Data = tmp_data;%×îÖÕblock
    Mask = tmp_mask;%×îÖÕblock mask
    
    blocks_all{1}=Data;
    mask_all{1}=Mask;
    blocks = 1;

    
else
    step = 5;
    blocks = floor((size(mask,3)-16)/step)+2;
    if mod((size(mask,3)-16),step)==0
        blocks=blocks-1;
    end
    for iii = 1:(blocks-1)
        Data = zeros(size(mask,1),size(mask,2),16);
        Mask = Data;
        Data = data(:,:,(1+(iii-1)*step):(16+(iii-1)*step));
        Mask = mask(:,:,(1+(iii-1)*step):(16+(iii-1)*step));
        blocks_all{end+1}=Data;
        mask_all{end+1}=Mask;
    end
    Data = zeros(size(mask,1),size(mask,2),16);
    Mask = Data;
    Data = data(:,:,end-15:end);
    Mask = mask(:,:,end-15:end);
    blocks_all{end+1}=Data;
    mask_all{end+1}=Mask;
end

end

