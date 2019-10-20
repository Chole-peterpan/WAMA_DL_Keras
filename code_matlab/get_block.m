function [blocks_all,masks_all,blocks] = get_block(data,mask,step,deepth)

blocks_all = {};
masks_all = {};
for i = 1:length(data)
    [blocks_all{end+1},masks_all{end+1},blocks] = get_block_from_tumor(data{i},mask{i},step,deepth);
end


