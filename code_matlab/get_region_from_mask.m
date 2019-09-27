function [tumor_mat,mask_mat,tumor_volum,or_size,location] = get_region_from_mask(data,mask,extend_pixel,square_flag)
%�ٳ�����
%ratio,����ѹ��������������ֹ���ѧϰʱ������������Ϊ0�򱣳�ԭ�������������ѹ����

disp('cuting...');
index = find(mask);

%Ĭ�Ϻ����ȡ������
if ~exist('square_flag', 'var') || isempty(square_flag)
    % opt_normalize ����Ϊ�գ����߲��Ա�������ʽ���ڣ�
    square_flag = true;
end


% cut
[I1,I2,I3] = ind2sub(size(mask),index);
d1_min = min(I1)-extend_pixel;
d1_max = max(I1)+extend_pixel;
d2_min = min(I2)-extend_pixel;
d2_max = max(I2)+extend_pixel;
d3_min = min(I3);
d3_max = max(I3);
if square_flag
   d1 = d1_max-d1_min;
   d2 = d2_max-d2_min;
   max_d = max(d1,d2);
   d1_max = d1_min+max_d;%
   d2_max = d2_min+max_d; 
end

location = [min(I1),max(I1),min(I2),max(I2),min(I3),max(I3)];
tumor_mat = data(d1_min:d1_max,  d2_min:d2_max,  d3_min:d3_max);
mask_mat =  mask(d1_min:d1_max,  d2_min:d2_max,  d3_min:d3_max);
or_size = size(tumor_mat);
tumor_volum = or_size(1)*or_size(2)*or_size(3);%���Ե�����������



end

