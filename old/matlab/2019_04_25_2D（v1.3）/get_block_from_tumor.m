function [blocks_all,mask_all,blocks] = get_block_from_tumor(data,mask)
blocks_all = {};
mask_all = {};
for iii = 1:size(data,3)
    blocks_all{end+1}=imresize(data(:,:,iii),[224,224],'bilinear');
    mask_all{end+1}=imresize(mask(:,:,iii),[224,224],'nearest');
end
blocks = size(data,3);
end

