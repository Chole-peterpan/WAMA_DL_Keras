function [out] = autoadd(matrix,vector)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
% �����ڽ�������ӵ���ά����

if isempty(matrix)
    % opt_normalize ����Ϊ�գ����߲��Ա�������ʽ���ڣ�
    out = vector;
else
    size_in = size(matrix);
    size_add = size(vector);
    if size_in(2) == size_add(2)
        out = [matrix;vector];
    elseif size_in(2) > size_add(2)
        out = [matrix;[vector,zeros(1,size_in(2)-size_add(2))]];
    else
        out = [[matrix,zeros(size_in(1),size_add(2) - size_in(2))];vector];
    end
    
end












end

