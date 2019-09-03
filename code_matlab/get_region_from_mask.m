function [tumor_mat,mask_mat,tumor_volum,or_size] = get_region_from_mask(data,mask,extend_pixel)
%�ٳ�����
%ratio,����ѹ��������������ֹ���ѧϰʱ������������Ϊ0�򱣳�ԭ�������������ѹ����
disp('cuting...');
index = find(mask);

% cut
[I1,I2,I3] = ind2sub(size(mask),index);
d1_min = min(I1)-extend_pixel;
d1_max = max(I1)+extend_pixel;
d2_min = min(I2)-extend_pixel;
d2_max = max(I2)+extend_pixel;
d3_min = min(I3);
d3_max = max(I3);







tumor_mat = data(d1_min:d1_max,  d2_min:d2_max,  d3_min:d3_max);
mask_mat =  mask(d1_min:d1_max,  d2_min:d2_max,  d3_min:d3_max);
or_size = size(tumor_mat);
tumor_volum = or_size(1)*or_size(2)*or_size(3);%���Ե�����������





% %resize
% if ratio == 0
%     tumor_mat = imresize3(data_cut,[280,280,or_size(3)],'cubic');
%     mask_mat = imresize3(mask_cut,[280,280,or_size(3)],'cubic');
% else
%     bili = (280/mean(or_size(1:2)))/ratio;%����ѹ��5���������ѧϰʱ���ֹ�������
%     z_length = floor(or_size(3)*bili);
%     
%     tumor_mat = imresize3(data_cut,[280,280,z_length],'cubic');
%     mask_mat = imresize3(mask_cut,[280,280,z_length],'cubic');
% end

end

