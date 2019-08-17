label = 1;%�Լ��趨һ��label

% 
pre = 0:0.001:1;
loss_eui_b1 = zeros(size(pre));%eui��ʧ
loss_eui_b2 = zeros(size(pre));%eui��ʧ
loss_eui_b4 = zeros(size(pre));%eui��ʧ
loss_eui_b6 = zeros(size(pre));%eui��ʧ
loss_Be = zeros(size(pre));%��Ԫ��������ʧ
for i = 1:1001
    loss_eui_b1(i)=EUIloss(pre(i),label,1);
    loss_eui_b2(i)=EUIloss(pre(i),label,2);
    loss_eui_b4(i)=EUIloss(pre(i),label,4);
    loss_eui_b6(i)=EUIloss(pre(i),label,6);
    loss_Be(i)=binary_cross_entropy(pre(i),label);
end
figure;subplot(1,3,1);
plot(pre,loss_eui_b1,'r');hold on;
plot(pre,loss_eui_b2,'g');hold on;
plot(pre,loss_eui_b4,'b');hold on;
plot(pre,loss_eui_b6,'m');hold on;
plot(pre,loss_Be,'k');title('label:1');hold on;
xlabel('prediction��Ԥ�����ֵ');
ylabel('loss');
legend({'EUIloss alpha:1','EUIloss alpha:2','EUIloss alpha:4','EUIloss alpha:6','binary cross entropy loss'},'Location','best');
 

%��ʾһ���ݶ�
diff_euiloss_b1 = abs(diff(loss_eui_b1));
diff_euiloss_b1(1:501)=diff_euiloss_b1(300);%����ͻ�䴦���ݶ�
diff_euiloss_b1(501:701)=diff_euiloss_b1(600);%����ͻ�䴦���ݶ�
diff_euiloss_b1(881:end)=diff_euiloss_b1(900);%����ͻ�䴦���ݶ�

diff_euiloss_b2 = abs(diff(loss_eui_b2));
diff_euiloss_b2(1:501)=diff_euiloss_b2(300);%����ͻ�䴦���ݶ�
diff_euiloss_b2(501:701)=diff_euiloss_b2(600);%����ͻ�䴦���ݶ�
diff_euiloss_b2(881:end)=diff_euiloss_b2(900);%����ͻ�䴦���ݶ�

diff_euiloss_b4 = abs(diff(loss_eui_b4));
diff_euiloss_b4(1:501)=diff_euiloss_b4(300);%����ͻ�䴦���ݶ�
diff_euiloss_b4(501:701)=diff_euiloss_b4(600);%����ͻ�䴦���ݶ�
diff_euiloss_b4(881:end)=diff_euiloss_b4(900);%����ͻ�䴦���ݶ�

diff_euiloss_b6 = abs(diff(loss_eui_b6));
diff_euiloss_b6(1:501)=diff_euiloss_b6(300);%����ͻ�䴦���ݶ�
diff_euiloss_b6(501:701)=diff_euiloss_b6(600);%����ͻ�䴦���ݶ�
diff_euiloss_b6(881:end)=diff_euiloss_b6(900);%����ͻ�䴦���ݶ�

diff_loss_Be = abs(diff(loss_Be));


subplot(1,3,2);
plot((0.001:0.001:1),diff_euiloss_b1,'r');hold on;
plot((0.001:0.001:1),diff_euiloss_b2,'g');hold on;
plot((0.001:0.001:1),diff_euiloss_b4,'b');hold on;
plot((0.001:0.001:1),diff_euiloss_b6,'m');hold on;
plot((0.001:0.001:1),diff_loss_Be,'k');title('label:1');
xlabel('prediction��Ԥ�����ֵ');
ylabel('derivative������');
legend({'EUIloss alpha:1','EUIloss alpha:2','EUIloss alpha:4','EUIloss alpha:6','binary cross entropy loss'},'Location','best');
title('label:0 , diff');

%������Ĺ۲�һ���ݶ�
qwe = (0.001:0.001:1);
subplot(1,3,3);
plot(qwe(100:end),diff_euiloss_b1(100:end),'r');hold on;
plot(qwe(100:end),diff_euiloss_b2(100:end),'g');hold on;
plot(qwe(100:end),diff_euiloss_b4(100:end),'b');hold on;
plot(qwe(100:end),diff_euiloss_b6(100:end),'m');hold on;
plot(qwe(100:end),diff_loss_Be(100:end),'k');title('label:0');
xlabel('prediction��Ԥ�����ֵ');
ylabel('derivative������');
legend({'EUIloss alpha:1','EUIloss alpha:2','EUIloss alpha:4','EUIloss alpha:6','binary cross entropy loss'},'Location','best');
title('label:0 , diff, Ϊ�˷���۲죬����ȥ��������ʾ');
%% 
label = 0;%�Լ��趨һ��label

% 
pre = 0:0.001:1;
loss_eui_b1 = zeros(size(pre));%eui��ʧ
loss_eui_b2 = zeros(size(pre));%eui��ʧ
loss_eui_b4 = zeros(size(pre));%eui��ʧ
loss_eui_b6 = zeros(size(pre));%eui��ʧ
loss_Be = zeros(size(pre));%��Ԫ��������ʧ
for i = 1:1001
    loss_eui_b1(i)=EUIloss(pre(i),label,1);
    loss_eui_b2(i)=EUIloss(pre(i),label,2);
    loss_eui_b4(i)=EUIloss(pre(i),label,4);
    loss_eui_b6(i)=EUIloss(pre(i),label,6);
    loss_Be(i)=binary_cross_entropy(pre(i),label);
end
figure;subplot(1,3,1);
plot(pre,loss_eui_b1,'r');hold on;
plot(pre,loss_eui_b2,'g');hold on;
plot(pre,loss_eui_b4,'b');hold on;
plot(pre,loss_eui_b6,'m');hold on;
plot(pre,loss_Be,'k');title('label:0');hold on;
xlabel('prediction��Ԥ�����ֵ');
ylabel('loss');
legend({'EUIloss alpha:1','EUIloss alpha:2','EUIloss alpha:4','EUIloss alpha:6','binary cross entropy loss'},'Location','best');
 

%��ʾһ���ݶ�
diff_euiloss_b1 = abs(diff(loss_eui_b1));
diff_euiloss_b1(500:1000)=diff_euiloss_b1(800);%����ͻ�䴦���ݶ�
diff_euiloss_b1(300:499)=diff_euiloss_b1(400);%����ͻ�䴦���ݶ�
diff_euiloss_b1(1:120)=diff_euiloss_b1(240);%����ͻ�䴦���ݶ�

diff_euiloss_b2 = abs(diff(loss_eui_b2));
diff_euiloss_b2(500:1000)=diff_euiloss_b2(800);%����ͻ�䴦���ݶ�
diff_euiloss_b2(300:499)=diff_euiloss_b2(400);%����ͻ�䴦���ݶ�
diff_euiloss_b2(1:120)=diff_euiloss_b2(240);%����ͻ�䴦���ݶ�

diff_euiloss_b4 = abs(diff(loss_eui_b4));
diff_euiloss_b4(500:1000)=diff_euiloss_b4(800);%����ͻ�䴦���ݶ�
diff_euiloss_b4(300:499)=diff_euiloss_b4(400);%����ͻ�䴦���ݶ�
diff_euiloss_b4(1:120)=diff_euiloss_b4(240);%����ͻ�䴦���ݶ�

diff_euiloss_b6 = abs(diff(loss_eui_b6));
diff_euiloss_b6(500:1000)=diff_euiloss_b6(800);%����ͻ�䴦���ݶ�
diff_euiloss_b6(300:499)=diff_euiloss_b6(400);%����ͻ�䴦���ݶ�
diff_euiloss_b6(1:120)=diff_euiloss_b6(240);%����ͻ�䴦���ݶ�

diff_loss_Be = abs(diff(loss_Be));


subplot(1,3,2);
plot((0.001:0.001:1),diff_euiloss_b1,'r');hold on;
plot((0.001:0.001:1),diff_euiloss_b2,'g');hold on;
plot((0.001:0.001:1),diff_euiloss_b4,'b');hold on;
plot((0.001:0.001:1),diff_euiloss_b6,'m');hold on;
plot((0.001:0.001:1),diff_loss_Be,'k');title('label:0');
xlabel('prediction��Ԥ�����ֵ');
ylabel('derivative������');
legend({'EUIloss alpha:1','EUIloss alpha:2','EUIloss alpha:4','EUIloss alpha:6','binary cross entropy loss'},'Location','best');
title('label:0 , diff');

%������Ĺ۲�һ���ݶ�
qwe = (0.001:0.001:1);
subplot(1,3,3);
plot(qwe(1:900),diff_euiloss_b1(1:900),'r');hold on;
plot(qwe(1:900),diff_euiloss_b2(1:900),'g');hold on;
plot(qwe(1:900),diff_euiloss_b4(1:900),'b');hold on;
plot(qwe(1:900),diff_euiloss_b6(1:900),'m');hold on;
plot(qwe(1:900),diff_loss_Be(1:900),'k');title('label:0');
xlabel('prediction��Ԥ�����ֵ');
ylabel('derivative������');
legend({'EUIloss alpha:1','EUIloss alpha:2','EUIloss alpha:4','EUIloss alpha:6','binary cross entropy loss'},'Location','best');
title('label:0 , diff, Ϊ�˷���۲죬����ȥ��������ʾ');








