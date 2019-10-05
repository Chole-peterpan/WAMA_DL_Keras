%% 根据s-id，抽取对应的1）loss，2）pre，3）pre-label，4）true-label，5）per_block_name，6）block_name






if s_id ~= 0
    name_index = 1;
    s_loss = block_loss(block_name(:,name_index) == s_id,:);
    s_pre = block_pre(block_name(:,name_index) == s_id,:);
    s_pre_label = block_pre_label(block_name(:,name_index) == s_id,:);
    s_true_label = block_true_label(block_name(:,name_index) == s_id,:);
    s_block_name = per_block_name(block_name(:,name_index) == s_id);
    s_block_name_mat = block_name(block_name(:,name_index) == s_id,:);
    
    if v_id == 0
        uqid_v_id = unique(s_block_name_mat(:,2));
        disp(['subject ',num2str(s_id),' have :',num2str(length(uqid_v_id)),' V',', and the v_ids are:']);
        disp(num2str(uqid_v_id));
    end
    if v_id == 0 && show_flag
        % 只有下一级id为0时候，才允许显示
        % plot block的loss
        figure;
        for ii = 1:length(s_block_name)
            plot(all_loss_iter,s_loss(ii,:)) ; hold on;
        end
        hold off;
        title('block loss');
        
        %loss的heatmap
        figure;
        imagesc(imresize(s_loss,size(s_loss)*scale_size,'nearest'));
        colormap(parula);
        colorbar('location','SouthOutside');
        ylabel('block name');
        xlabel('iters');
        
        ytick = floor(1:(size(s_loss,1)*scale_size/length(s_block_name)):size(s_loss,1)*scale_size);
        set(gca, 'yTick', ytick);
        set(gca,'YTickLabel',s_block_name)
        
        xtick = floor(1:(size(s_loss,2)*scale_size/length(all_loss_iter)):size(s_loss,2)*scale_size);
        set(gca, 'xTick', xtick);
        set(gca,'XTickLabel',all_loss_iter)
        
        title('block loss heatmap');
        
        
        %pre的heatmap
        figure;
        imagesc(imresize(s_pre,size(s_pre)*scale_size,'nearest'));
        colormap(parula);
        colorbar('location','SouthOutside');
        ylabel('block name');
        xlabel('iters');
        
        ytick = floor(1:(size(s_pre,1)*scale_size/length(s_block_name)):size(s_pre,1)*scale_size);
        set(gca, 'yTick', ytick);
        set(gca,'YTickLabel',s_block_name)
        
        xtick = floor(1:(size(s_pre,2)*scale_size/length(all_loss_iter)):size(s_pre,2)*scale_size);
        set(gca, 'xTick', xtick);
        set(gca,'XTickLabel',all_loss_iter)
        
        title('block pre heatmap');
        
        
        
        % 针对某个CT，或者肿瘤的acc显示（暂时先不考虑，后续跑其他数据时候再添加对应代码）
        
        
    end
end


if v_id ~=0
    name_index = 2;
    tmp_id = v_id;
    s_loss = s_loss(s_block_name_mat(:,name_index) == tmp_id,:);
    s_pre = s_pre(s_block_name_mat(:,name_index) == tmp_id,:);
    s_pre_label = s_pre_label(s_block_name_mat(:,name_index) == tmp_id,:);
    s_true_label = s_true_label(s_block_name_mat(:,name_index) == tmp_id,:);
    s_block_name = s_block_name(s_block_name_mat(:,name_index) == tmp_id);
    s_block_name_mat = s_block_name_mat(s_block_name_mat(:,name_index) == tmp_id,:);
    
    
    if m_id == 0
        uqid_m_id = unique(s_block_name_mat(:,3));
        disp(['subject_id:',num2str(s_id),' V-CT_id:',num2str(v_id),' have :',num2str(length(uqid_m_id)),' masks/tumors',', and the m_ids are:']);
        disp(num2str(uqid_m_id));
    end
    
    if m_id == 0 && show_flag
        % 只有下一级id为0时候，才允许显示
        % plot block的loss
        figure;
        for ii = 1:length(s_block_name)
            plot(all_loss_iter,s_loss(ii,:)) ; hold on;
        end
        hold off;
        title('block loss');
        
        %loss的heatmap
        figure;
        imagesc(imresize(s_loss,size(s_loss)*scale_size,'nearest'));
        colormap(parula);
        colorbar('location','SouthOutside');
        ylabel('block name');
        xlabel('iters');
        
        ytick = floor(1:(size(s_loss,1)*scale_size/length(s_block_name)):size(s_loss,1)*scale_size);
        set(gca, 'yTick', ytick);
        set(gca,'YTickLabel',s_block_name)
        
        xtick = floor(1:(size(s_loss,2)*scale_size/length(all_loss_iter)):size(s_loss,2)*scale_size);
        set(gca, 'xTick', xtick);
        set(gca,'XTickLabel',all_loss_iter)
        
        title('block loss heatmap');
        
        
        %pre的heatmap
        figure;
        imagesc(imresize(s_pre,size(s_pre)*scale_size,'nearest'));
        colormap(parula);
        colorbar('location','SouthOutside');
        ylabel('block name');
        xlabel('iters');
        
        ytick = floor(1:(size(s_pre,1)*scale_size/length(s_block_name)):size(s_pre,1)*scale_size);
        set(gca, 'yTick', ytick);
        set(gca,'YTickLabel',s_block_name)
        
        xtick = floor(1:(size(s_pre,2)*scale_size/length(all_loss_iter)):size(s_pre,2)*scale_size);
        set(gca, 'xTick', xtick);
        set(gca,'XTickLabel',all_loss_iter)
        
        title('block pre heatmap');
        
        
        
        % 针对某个CT，或者肿瘤的acc显示（暂时先不考虑，后续跑其他数据时候再添加对应代码）
        
    end
    
    
    
end


if m_id ~= 0
    name_index = 3;
    tmp_id = m_id;
    s_loss = s_loss(s_block_name_mat(:,name_index) == tmp_id,:);
    s_pre = s_pre(s_block_name_mat(:,name_index) == tmp_id,:);
    s_pre_label = s_pre_label(s_block_name_mat(:,name_index) == tmp_id,:);
    s_true_label = s_true_label(s_block_name_mat(:,name_index) == tmp_id,:);
    s_block_name = s_block_name(s_block_name_mat(:,name_index) == tmp_id);
    s_block_name_mat = s_block_name_mat(s_block_name_mat(:,name_index) == tmp_id,:);
    
    
    uqid_b_id = unique(s_block_name_mat(:,4));
    disp(['subject_id:',num2str(s_id),' V-CT_id:',num2str(v_id),' M_id:',num2str(m_id),' have :',num2str(length(uqid_b_id)),' blocks',', and the b_ids are:']);
    disp(num2str(uqid_b_id));
    
    
    if show_flag
        % 只有下一级id为0时候，才允许显示
        % plot block的loss
        figure;
        for ii = 1:length(s_block_name)
            plot(all_loss_iter,s_loss(ii,:)) ; hold on;
        end
        hold off;
        title('block loss');
        
        %loss的heatmap
        figure;
        imagesc(imresize(s_loss,size(s_loss)*scale_size,'nearest'));
        colormap(parula);
        colorbar('location','SouthOutside');
        ylabel('block name');
        xlabel('iters');
        
        ytick = floor(1:(size(s_loss,1)*scale_size/length(s_block_name)):size(s_loss,1)*scale_size);
        set(gca, 'yTick', ytick);
        set(gca,'YTickLabel',s_block_name)
        
        xtick = floor(1:(size(s_loss,2)*scale_size/length(all_loss_iter)):size(s_loss,2)*scale_size);
        set(gca, 'xTick', xtick);
        set(gca,'XTickLabel',all_loss_iter)
        
        title('block loss heatmap');
        
        
        %pre的heatmap
        figure;
        imagesc(imresize(s_pre,size(s_pre)*scale_size,'nearest'));
        colormap(parula);
        colorbar('location','SouthOutside');
        ylabel('block name');
        xlabel('iters');
        
        ytick = floor(1:(size(s_pre,1)*scale_size/length(s_block_name)):size(s_pre,1)*scale_size);
        set(gca, 'yTick', ytick);
        set(gca,'YTickLabel',s_block_name)
        
        xtick = floor(1:(size(s_pre,2)*scale_size/length(all_loss_iter)):size(s_pre,2)*scale_size);
        set(gca, 'xTick', xtick);
        set(gca,'XTickLabel',all_loss_iter)
        
        title('block pre heatmap');
        
        
        
        % 针对某个CT，或者肿瘤的acc显示（暂时先不考虑，后续跑其他数据时候再添加对应代码）
        
    end
    
    
    
end











