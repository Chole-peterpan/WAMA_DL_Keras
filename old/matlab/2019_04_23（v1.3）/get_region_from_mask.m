function [tumor_mat,mask_mat,tumor_size,or_size] = get_region_from_mask(data,mask,ratio)
%�ٳ�����
%ratio,����ѹ��������������ֹ���ѧϰʱ������������Ϊ0�򱣳�ԭ�������������ѹ����
disp('cuting...');
index = find(mask);

% cut
[I1,I2,I3] = ind2sub(size(mask),index);
data_cut = data(min(I1):max(I1),min(I2):max(I2),min(I3):max(I3));
mask_cut = mask(min(I1):max(I1),min(I2):max(I2),min(I3):max(I3));
or_size = size(data_cut);
tumor_size = or_size(1)*or_size(2)*or_size(3);%���Ե�����������

%resize
if ratio == 0
    tumor_mat = imresize3(data_cut,[280,280,or_size(3)],'cubic');
    mask_mat = imresize3(mask_cut,[280,280,or_size(3)],'cubic');
else
    bili = (280/mean(or_size(1:2)))/ratio;%����ѹ��5���������ѧϰʱ���ֹ�������
    z_length = floor(or_size(3)*bili);
    
    tumor_mat = imresize3(data_cut,[280,280,z_length],'cubic');
    mask_mat = imresize3(mask_cut,[280,280,z_length],'cubic');
end

end

