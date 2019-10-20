function [out] = autoadd(matrix,vector)
%UNTITLED 此处显示有关此函数的摘要
% 仅用于将向量添加到二维数组

if isempty(matrix)
    % opt_normalize 参数为空，或者不以变量的形式存在；
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

