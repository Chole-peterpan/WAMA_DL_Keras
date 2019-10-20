% clear;close all;
path='G:\ÌôÕ½±­£¨ÐÂ£©\wang_alexnet_e-5\fold1';
finf = dir([path,'\*.txt']);

n = length(finf);
data = cell(n,1);
for k=1:n
    filename = [path,filesep,finf(k).name];
    data{k} = importdata(filename);
end


acc = zeros(n,1); 
for kk=1:n
   tmpstr=char(data{kk});
   index_point=find(tmpstr=='.');
   acc(kk)= str2double(tmpstr(index_point(3):end));
end

index_maxacc=find(acc==max(acc));
best_acc=max(acc);
final = cell(length(index_maxacc),2);
for k=1:length(index_maxacc)
    final{k,1} = finf(index_maxacc(k)).name;
    final{k,2} = (data{index_maxacc(k)});
end















