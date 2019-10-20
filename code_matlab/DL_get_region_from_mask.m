function [segment_mat,mask_mat,segment_size,or_size] = DL_get_region_from_mask(data,mask,ratio)
%�ٳ�����
%ratio,����ѹ��������������ֹ���ѧϰʱ������������Ϊ0�򱣳�ԭ�������������ѹ����
disp('region growing...')
%���ò���===========================================
slice_index = [];   %ROI�����Ĳ���
mask_ = [];     %��ת���mask��ʹ��������Z
data_ = [];     %��mask_ͬ����ת
roi_ = [];     
%�������������ת��
m_1 = mask;
m_2 = permute(mask,[3,1,2]);
m_3 = permute(mask,[2,3,1]);
%ȷ��ROI������һ������==================================
[aera,dim] = max([max(sum(sum(m_1))),max(sum(sum(m_2))),max(sum(sum(m_3)))]);

if dim == 1     %����1
    slice_index = find(sum(sum(m_1)));  %��������
    roi_ = m_1(:,:,slice_index);    %������ROI
    mask_ = m_1;    
    data_ = data;
elseif dim == 2     %����2
    slice_index = find(sum(sum(m_2)));
    roi_ = m_2(:,:,slice_index);
    mask_ = m_2;
    data_ = permute(data,[3,1,2]);
elseif dim == 3     %����3
    slice_index = find(sum(sum(m_3)));
    roi_ = m_3(:,:,slice_index);
    mask_ = m_3;
    data_ = permute(data,[2,3,1]);
end

slice_num = length(slice_index);
disp('the Number of Slices is:')
disp(slice_num)
%һ�����е����=(�ѷ�ת������z����)=====================================
if slice_num == 1
    %��Ψһ���ѡ���������������ƶ�����(��ž��ƶ����ΰ뾶)
    diameter = max(max(sum(roi_)),max(sum(roi_,2)));
    distance = floor(diameter/2)+1; %���Ӷ���
    index = find(roi_);
    [I1,I2] = ind2sub(size(roi_),index);
    roi_cut = roi_(min(I1):max(I1),min(I2):max(I2));    %jiukankan
    % Cut
    data_cut = data_(min(I1):max(I1),min(I2):max(I2),slice_index-distance:slice_index+distance);
    data_cut = permute(data_cut,[1 3 2]);   %���תΪ����
    mask_cut = mask_(min(I1):max(I1),min(I2):max(I2),slice_index-distance:slice_index+distance);
    mask_cut = permute(mask_cut,[1 3 2]);   %���תΪ����
    or_size = size(data_cut);
    segment_size = or_size(1)*or_size(2)*or_size(3);    %���Ե�����������
%����ͬά�����=(�ѷ�ת������z����)=========================================
elseif slice_num > 1
    index = find(mask_);
    % cut
    [I1,I2,I3] = ind2sub(size(mask_),index);
    data_cut = data_(min(I1):max(I1),min(I2):max(I2),min(I3):max(I3));
    mask_cut = mask_(min(I1):max(I1),min(I2):max(I2),min(I3):max(I3));
    or_size = size(data_cut);
    segment_size = or_size(1)*or_size(2)*or_size(3);%���Ե�����������
end
%resize====================================================
if ratio == 0
    segment_mat = imresize3(data_cut,[280,280,or_size(3)],'cubic');
    mask_mat = imresize3(mask_cut,[280,280,or_size(3)],'cubic');
else
    bili = (280/mean(or_size(1:2)))/ratio;%����ѹ��5���������ѧϰʱ���ֹ�������
    z_length = floor(or_size(3)*bili);
    
    segment_mat = imresize3(data_cut,[280,280,z_length],'cubic');
    mask_mat = imresize3(mask_cut,[280,280,z_length],'cubic');
end

end

