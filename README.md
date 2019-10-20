<font size=5> **环境和依赖包建议**</font>

**matlab：**

<font size=3>`MATLAB 2017` 或更高版本，低于2017将无法执行脚本`STEP1.m`到`STEP5.m`

[`SPM8 toolbox` ](http://www.fil.ion.ucl.ac.uk/spm/software/spm8/) </font>

**python：**

<font size=3>`keras 2.1.1` （高版本可能不支持过GPU数据并行）

`tensorflow 1.3.0` （高版本可能无法使用多GPU数据并行模式）

`prettytable` 任意版本</font>

# 1.代码组成
 - <font size=5>预处理以及病例信息统计代码：主要包括5个<font color=#008000 >matlab</font>代码，分别为：</font>
  
     `STEP1.m` : <font size=3>根据roi从NIFTI文件中提取病灶，并转换为mat文件，同时将所有病例的roi信息整合以便统计</font>
     
     `STEP2.m` : <font size=3>对提取到病灶进行分块（patch）</font>
     
     `STEP3.m` : <font size=3>将patch保存为H5文件</font>
     
     `STEP4.m` : <font size=3>将patch进行扩增并保存为H5文件</font>
     
     `STEP5.m` : <font size=3>对准备好的数据分折，并保存分折信息以供后续交叉验证</font>
     
     `CHECK.m` : <font size=3>沿Z轴分层显示`STEP2.m` `STEP3.m` `STEP4.m` 生成的patch，以供检查数据</font>
     
     `STATISTIC.m` : <font size=3>统计并绘图显示各病例的扫描段数、肿瘤数量、肿瘤位置、肿瘤长宽高及长宽高比、各CT分辨率（voxel size），以及显示扩增前后数量对比，可在`STEP1.m` `STEP2.m` `STEP4.m`后使用</font>
     
 -  <font size=5>网络训练与监测代码：主要包括1个<font color= Blue >python</font>代码和1个<font color=#008000 >matlab</font>代码，分别为：</font>
  
     `MAIN.py` :  <font size=3>网络训练代码，在训练过程中会同时测试和验证</font>
     
     `WATCHer.m` :  <font size=3>网络监测，可实时监测网络学习率，以及各数据集loss、acc等指标</font>
     
	 `proWATCHer.m` ：<font size=3>进一步详细显示各数据集预测结果。可指定某个病人，或某个病人的某个patch以单独观察</font>
	      
 -  <font size=5>训练结果汇总代码：主要包括2个<font color=#008000 >matlab</font>代码，分别为：</font>
  
      `STEP6.m` :  <font size=3>交叉验证单折结果汇总代码</font>
      
      `STEP7.m` :  <font size=3>汇总所有折结果，并做出混淆矩阵，如果为二分类问题则补充AUC曲线</font>


# 2.实验过程
## 1.数据和环境准备

<font size=4>**将病人的CT数据以及roi数据按照以下格式命名，之后将所有CT与ROI文件放在同一文件夹下：**</font>

CT数据命名为：  s<font color=red>病人ID</font>_v<font color=red>扫描段</font>.nii

ROI数据命名为：s<font color=red>病人ID</font>_v<font color=red>扫描段</font>_m<font color=red>ROI序号</font>.nii

<font size=3>**ps**: 扫描段指的是：一个病人可能分多段扫描，比如先扫上腹部，再扫描下腹部</font>

<font size=3>**eg**.  s1_v1.nii就是1号病人的第一个扫描段，s1_v1_m1即对应s1_v1.nii；s1_v2.nii就是1号病人的第一个扫描段，s1_v2_m1即对应s1_v2.nii。下图中为已命名好的例子</font>

<font color=red>***如果有其他模态/序列的数据，则以同样命名格式存放在另一文件夹，并保证各模态文件夹中数据文件名相同***</font>

![enter image description here](https://github.com/WAMAWAMA/WAMA_DL_Keras/blob/v3_beta/pic/1571545292%281%29.jpg?raw=true)

<font size=4>**将病人的label以表格形式储存，表格内容要求为：**</font>

 - <font size=3>第一行为变量名</font>
 - <font size=3>第一列为病人序号（ID），无需从0开始，但必须全部为数字，可为int也可为double
其余列为label，各组label可包含两类或多类，且必须从0开始以连续正整数命名（如二分类为0，1，三分类为0，1，2）</font>

<font size=3>**例子如下图**</font>

![enter image description here](https://github.com/WAMAWAMA/WAMA_DL_Keras/blob/v3_beta/pic/73JA@ZV$$S_%291Q7BBMBOP2W.png?raw=true)

<font size=4>**matlab并行设置：**</font>

<font size=3>若想加速matlab预处理速度，则需在matlab的parallel中的preference将parallel pool设置为电脑CPU核心数-1（或-2），之后取消勾选Shut down</font>

![enter image description here](https://github.com/WAMAWAMA/WAMA_DL_Keras/blob/v3_beta/pic/URYFT%25Y%5D%5BFPT5N%5B%25%5DBY0T_4.png?raw=true)
![enter image description here](https://github.com/WAMAWAMA/WAMA_DL_Keras/blob/v3_beta/pic/45@~1%7D3SFIPO%7D$Z0%5B6CL7%299.png?raw=true)

## 2.数据预处理
### 第一步：运行STEP1.m，读取NIFTI格式的CT数据和roi数据，将roi内组织提取出并保存为mat，同时将所有病人信息以结构体数组的形式保存在savepath的subject文件夹中，运行STEP1.m主要设置参数如下：

```matlab
%% 设置路径参数
% 模态or序列1
data_path{end+1} = 'G:\estdata\data\0nii'; %数据存放的路径
data_window{end+1} = [-25,285]; %针对CT数据设置的窗宽窗位，数值为[窗位-0.5窗宽，窗位+0.5窗宽]
% 模态or序列2（如果没有则不加）
% data_path{end+1} = 'G:\estdata\data\0nii';
% data_window{end+1} = [-10,205];
% 模态or序列3
% data_path{end+1} = 'G:\estdata\data\0nii';
% data_window{end+1} = [-400,805];
% label文件完整路径
label_file = 'G:\estdata\data\label.xlsx';
% 预处理结果保存路径
save_path = 'G:\estdata\1mat';

%% 设定其他参数
% 是否重采样体素size
resample_flag = true; 
% roi外扩的大小（单位为mm）
extend_length = 0; % 如果为0则不外扩，不可为负数
% 空间体素重采样的目标size（单位为mm）
adjust_voxelsize = [0.5,0.5,1.0]; 
   ``` 

### 第二步：运行STEP2.m，将STEP1输出的感兴趣区域分块（block 或 patch），这里的分块只沿着Z轴进行分块。运行STEP2.m主要设置参数如下：

```matlab
%% 设置路径参数
mat_path =       'G:\estdata\1mat'; %STEP1输出结果的文件夹，即STEP1的save_path 
block_savepath = 'G:\estdata\2block';  %分块结果保存文件夹

%% 设置其他参数
step = 2; % 分块的滑动步长
deepth = 20;% 分块的层厚，如果为0，则不分块
   ``` 

### 第三步：运行STEP3.m，将STEP2输出的block储存到同名H5文件中。因为各个病人的block之间横截面大小不同，所以需要规定统一的输出大小。将block转换为固定大小共有以下4种模式：
 - mode 1 ：直接3D resize到输出size
 - mode 2：0矩阵容器居中 （注意，这个模式下目标形状的dim3要和block形状的dim3相同）, 如果block横截面大于输出形状横截面，则reshape以适应容器
 - mode 3：直接剪裁&padding （当目标维度小于原始维度则剪裁，当目标维度大于原始维度则padding），空间原点为matlab空间坐标原点（1，1，1），所以叫做直接剪裁
 - mode 4：与mode3类似，但空间原点为block中心

运行STEP2.m主要设置参数如下：
```matlab
%% 设置路径参数
mat_path =       'G:\estdata\1mat'; %STEP1输出结果的文件夹，即STEP1的save_path 
block_savepath = 'G:\estdata\2block';  %分块结果保存文件夹

%% 设置其他参数
step = 2; % 分块的滑动步长
deepth = 20;% 分块的层厚，如果为0，则不分块
   ``` 
### 第四步：运行STEP4.m，将STEP2输出的block**扩增**并储存到同名H5文件中。与STEP3不同的是：
<font size=3>（1）STEP4会根据指定组的label扩增所有block，一般会使该组label对应的**各类block扩增数量相等**，且每类中**各样本block扩增数量相等**，以达到类别与样本两个层次的分布平衡。需要设置的参数主要有两个：</font>
```matlab 
%总扩增数量
augdict.aug_num = 300;
%以第几组label的分类来进行平衡
augdict.balance_label_index = 1;
%平衡时候各类的比例（label值从小到大），全为1则类别平衡，也可以指定非平衡的比例
augdict.a_b_ratio = [1,1,1];%最终各类扩增数量比，例如ABC三类最终扩增数量比例为1:1:1，则达到类别平衡
```
<font size=3>（2）STEP4会对同一个block数据随机处理多次（类似于实时增强）以达到扩增数据的目的。所有处理均为2D处理且在横截面进行，可用的扩增手段为：</font>
 - <font size=3> 随机剪裁</font>
 -  <font size=3>随即拉伸</font>
 -  <font size=3>对比度调整</font>
 -  <font size=3>左右翻转</font>
 -  <font size=3>上下翻转</font>
 -  <font size=3>旋转</font>

<font size=3>每种手段均可通过指定执行的几率，如百分之3几率执行旋转，则</font>
```matlab 
augdict.rotation.p = 0.3;
```
<font size=3>具体的扩增参数设置请见`aug43D.m`</font>



### 第5步运行：STEP5.m，对数据集分折并保存分折信息到txt，以供接下来训练网络时读取：
分折时会保证每类样本在各折间均匀分布，且分折时以病人为单位分折。参数设置如下
```matlab
%% 参数设置
% 分3折折数(大于2)
K = 3;
% label序号(即按照第几组label来分折)
label_index = 1;
%% 路径设置
block_data = 'G:\estdata\2block'; % STEP2的输出路径
or_data = 'G:\estdata\3or'; % STEP3的输出路径
aug_data = 'G:\estdata\4aug'; % STEP4的输出路径
savepath = 'G:\estdata\5CV'; % 分折信息存放路径

%% 其他参数设置
% 是否使用已有的分折信息，如果已有分折，则可直接根据已有分折信息进行分折（如果没有则false）
have_folder = false;
% 已有分折信息路径
folder_file = 'H:\@data_NENs_recurrence\PNENs\data\flow1\5CV\mat_folder.mat';

   ``` 



