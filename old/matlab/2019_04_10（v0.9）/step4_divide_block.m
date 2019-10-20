clc;
clear;
mat_path = 'G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@data\v\3cut';
mat_savepath = 'G:\@data_zhuanyi_zhongshan\PNENs_wang_dl\@data\v\4block';


filename_list = dir(strcat(mat_path,filesep,'*.mat'));
for ii = 1:length(filename_list)
    filename = filename_list(ii,1).name;
    workspaces = load(strcat(mat_path,filesep,filename));
    data = workspaces.Data_cu;
    mask = workspaces.mask_cu;

    if size(mask,3)<16
        tmp_data = zeros(size(mask,1),size(mask,2),16);
        tmp_mask = tmp_data;
        tmp_data(:,:,1:size(mask,3)) = data;
        tmp_mask(:,:,1:size(mask,3)) = mask;
        Data = tmp_data;
        Mask = tmp_mask;
        blocks = 1;
        save(strcat(mat_savepath,filesep,filename(1:end-4),'_1'),'Data','Mask','blocks');
        
    else
        step = 5;
        blocks = floor((size(mask,3)-16)/step)+2;
        if mod((size(mask,3)-16),step)==0
           blocks=blocks-1; 
        end
        for iii = 1:(blocks-1)
            Data = zeros(size(mask,1),size(mask,2),16);
            Mask = Data;
            Data = data(:,:,(1+(iii-1)*step):(16+(iii-1)*step));
            Mask = mask(:,:,(1+(iii-1)*step):(16+(iii-1)*step));
            save(strcat(mat_savepath,filesep,filename(1:end-4),'_',num2str(iii)),'Data','Mask','blocks');
        end
        Data = zeros(size(mask,1),size(mask,2),16);
        Mask = Data;
        Data = data(:,:,end-15:end);
        Mask = mask(:,:,end-15:end);
        save(strcat(mat_savepath,filesep,filename(1:end-4),'_',num2str(blocks)),'Data','Mask','blocks');
    end
end



% % check
% for iii=1:size(Mask,3)
%     imshowpair(Data(:,:,iii), Data(:,:,iii), 'mon');
%     pause();
% end
% % 记得储存block数，从而确定扩增数量