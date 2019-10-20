%% 人工检查
%% 初始化
initialization;

%% 设定文件的路径
file_path{2} = 'G:\estdata\3or\s4_v1_m1_b2.h5';
file_path{1} = 'G:\estdata\4aug\s4_v1_m1_b2_e4.h5';

%% 初始化
data_type = {};
data = {};
label = {};

%% 加载图片
% 处理path
for n = 1:length(file_path)
    data_type{n} = split(file_path{n},'.');
    data_type{n} = data_type{n}{end};
    if strcmp(data_type{n} ,'mat')
        workspace = load(file_path{n});
        data{n} = workspace.block;
        label{n} = workspace.label;
    else
        fileInfo = h5info(file_path{n});
        Datasets = {fileInfo.Datasets.Name};
        mode = 0;
        for i = 1:length(Datasets)
            if strcmp(Datasets{i}(1:4),'mode')
                mode = mode+1;
            end
        end
        data{n} = {};
        for i =1:mode
            data{n}{end+1} = h5read(file_path{n},['/mode',num2str(i)]);
        end
        label{n} = h5read(file_path{n},'/label');
    end
end
%% 显示
figure;
for i = 1:size(data{1}{1},3)
    for ii = 1:length(data{1})
        subplot(2,length(data{1}),ii);
        imshow(data{1}{ii}(:,:,i),[0,1]);title(['datatype:',data_type{1},'mode:',num2str(ii),'layer',num2str(i)]);
        subplot(2,length(data{1}),ii+length(data{1}));
        imshow(data{2}{ii}(:,:,i),[0,1]);title(['datatype:',data_type{2},'mode:',num2str(ii),'layer',num2str(i)]);
    end
    pause(0.05); 
end






