function [segment_mat,mask_mat,segment_size,or_size] = DL_get_region_from_mask(data,mask,ratio)
%抠出函数
%ratio,纵轴压缩比例，用来防止深度学习时候输入过大，如果为0则保持原纵轴层数不进行压缩。
disp('region growing...')
%设置参数===========================================
slice_index = [];   %ROI勾画的层数
mask_ = [];     %旋转后的mask，使肠道走向Z
data_ = [];     %与mask_同步旋转
roi_ = [];     
%尝试三个方向的转法
m_1 = mask;
m_2 = permute(mask,[3,1,2]);
m_3 = permute(mask,[2,3,1]);
%确定ROI画在哪一个面上==================================
[aera,dim] = max([max(sum(sum(m_1))),max(sum(sum(m_2))),max(sum(sum(m_3)))]);

if dim == 1     %方向1
    slice_index = find(sum(sum(m_1)));  %勾画层数
    roi_ = m_1(:,:,slice_index);    %勾画的ROI
    mask_ = m_1;    
    data_ = data;
elseif dim == 2     %方向2
    slice_index = find(sum(sum(m_2)));
    roi_ = m_2(:,:,slice_index);
    mask_ = m_2;
    data_ = permute(data,[3,1,2]);
elseif dim == 3     %方向3
    slice_index = find(sum(sum(m_3)));
    roi_ = m_3(:,:,slice_index);
    mask_ = m_3;
    data_ = permute(data,[2,3,1]);
end

slice_num = length(slice_index);
disp('the Number of Slices is:')
disp(slice_num)
%一面纵切的情况=(已翻转，整体z走向)=====================================
if slice_num == 1
    %把唯一面框选出来，计算上下移动距离(大概就移动肠段半径)
    diameter = max(max(sum(roi_)),max(sum(roi_,2)));
    distance = floor(diameter/2)+1; %随便加多少
    index = find(roi_);
    [I1,I2] = ind2sub(size(roi_),index);
    roi_cut = roi_(min(I1):max(I1),min(I2):max(I2));    %jiukankan
    % Cut
    data_cut = data_(min(I1):max(I1),min(I2):max(I2),slice_index-distance:slice_index+distance);
    data_cut = permute(data_cut,[1 3 2]);   %最后转为上下
    mask_cut = mask_(min(I1):max(I1),min(I2):max(I2),slice_index-distance:slice_index+distance);
    mask_cut = permute(mask_cut,[1 3 2]);   %最后转为上下
    or_size = size(data_cut);
    segment_size = or_size(1)*or_size(2)*or_size(3);    %粗略地算出肠段体积
%三面同维的情况=(已翻转，肠道z走向)=========================================
elseif slice_num > 1
    index = find(mask_);
    % cut
    [I1,I2,I3] = ind2sub(size(mask_),index);
    data_cut = data_(min(I1):max(I1),min(I2):max(I2),min(I3):max(I3));
    mask_cut = mask_(min(I1):max(I1),min(I2):max(I2),min(I3):max(I3));
    or_size = size(data_cut);
    segment_size = or_size(1)*or_size(2)*or_size(3);%粗略地算出肠段体积
end
%resize====================================================
if ratio == 0
    segment_mat = imresize3(data_cut,[280,280,or_size(3)],'cubic');
    mask_mat = imresize3(mask_cut,[280,280,or_size(3)],'cubic');
else
    bili = (280/mean(or_size(1:2)))/ratio;%纵轴压缩5倍，以深度学习时候防止输入过大
    z_length = floor(or_size(3)*bili);
    
    segment_mat = imresize3(data_cut,[280,280,z_length],'cubic');
    mask_mat = imresize3(mask_cut,[280,280,z_length],'cubic');
end

end

