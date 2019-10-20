function [blocks_all,mask_all,blocks] = get_block_from_tumor(data,mask,step,deepth)

blocks_all = {};
mask_all = {};
%deepthΪ0ʱ��ԭ�ⲻ���ķ��ؼ���
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
        Data = tmp_data;%����block
        Mask = tmp_mask;%����block mask
        
        blocks_all{1}=Data;
        mask_all{1}=Mask;
        blocks = 1;
    else
        blocks = floor((size(mask,3)-deepth)/step)+2;
        % ������������������ȡ��ȫ���㣬��-1����Ϊ�������������ζ�Ҫ�Ӻ���ǰ����ȡһ��block
        if mod((size(mask,3)-deepth),step)==0
            blocks=blocks-1;
        end
        % ��ʼ����
        for iii = 1:(blocks-1)
            Data = zeros(size(mask,1),size(mask,2),deepth);
            Mask = Data;
            Data = data(:,:,(1+(iii-1)*step):(deepth+(iii-1)*step));
            Mask = mask(:,:,(1+(iii-1)*step):(deepth+(iii-1)*step));
            blocks_all{end+1}=Data;
            mask_all{end+1}=Mask;
        end
        
        % �Ӻ���ǰ����ȡһ��block
        Data = zeros(size(mask,1),size(mask,2),deepth);
        Mask = Data;
        Data = data(:,:,end-(deepth-1):end);
        Mask = mask(:,:,end-(deepth-1):end);
        blocks_all{end+1}=Data;
        mask_all{end+1}=Mask;
    end
end

end





















