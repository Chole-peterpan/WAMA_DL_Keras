## 本层次代码分为
from base_module.Resnet3D import resnet3D
from base_function.run import run
from base_function._function import *
from base_function.lr_method import *
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
file_sep = os.sep

# 设置路径和参数 ============================================================================
aug_data_path = '/media/root/老王3号/newnwenwennw/4aug'
or_data_path = '/media/root/老王3号/newnwenwennw/3or'
CV_file_path = '/media/root/老王3号/newnwenwennw/5CV'
foldname = '1'# 注意修改!!!
task_name = 'fold3'  # 任务名称,自己随便定义# 注意修改!!!
Result_save_Path = r'/media/root/老王3号/newnwenwennw/test'# 注意修改!!!
label_index = 1
data_loader = load_data_from_h5
loss_func = categorical_crossentropy
optimizer = Adam(lr = 1e-4)
train_batchsize = 15
test_batchsize = 3
GPU_index = "0,1,2"
train_epoch = 1000
printer = color_printer
model = resnet3D


# ===================================================================================
or_train_list = get_filelist_fromTXT(or_data_path, CV_file_path+file_sep+'fold_'+foldname+'_or_train.txt')
aug_train_list = get_filelist_fromTXT(aug_data_path, CV_file_path+file_sep+'fold_'+foldname+'_aug_train.txt')
or_test_list = test_list =get_filelist_fromTXT(or_data_path,CV_file_path+file_sep+'fold_'+foldname+'_or_test.txt')
or_ver_list = get_filelist_fromTXT(or_data_path,CV_file_path+file_sep+'fold_'+foldname+'_or_verify.txt')

# 先读一个数据,然后获得data_input_shape和label_shape
a, b, data_input_shape, label_shape, c = load_data_from_h5(or_test_list[0], label_index)

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_index
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# 构建model
model = model(classes=label_shape[0], inputshape=data_input_shape)


# 如果使用多于1块的GPU则数据并行
if len(GPU_index) != 1:
    model = multi_gpu_model(model, gpus=len(GPU_index.split(',')))
model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

# 学习率寻优:这一步之后,需要手动输入最佳lr范围
# random.shuffle(aug_train_list)
# finder_loss, finder_loss_smooth, finder_lr =  lr_finder(model = model,
#               data_loader = data_loader,
#               label_index = label_index,
#               tmp_weight_path = Result_save_Path,
#               filelist = aug_train_list+or_train_list,
#               batchsize = train_batchsize,
#               iter = 300,
#               beta = 0.98,
#               inc_mode = 'mult',
#               lr_low=1e-12,
#               lr_high=10,
#               show_flag = True)
# 手动设置
# 键盘手动输入lr_high和low
# keyboard_input = input(r'pls enter lr_high and lr_low')  # exp.   1e-6, 1e-11
# lr_high, lr_low = [float(i) for i in (keyboard_input.split(',')[:])]
lr_high, lr_low = [1e-4,1e-9]
lr_sgdr = lr_mod_cos(epoch_file_size=len(or_train_list) + len(aug_train_list),
                     batchsize=train_batchsize,
                     lr_high=lr_high,
                     lr_low=lr_low,
                     warmup_epoch=5,
                     loop_step=[1, 2, 4, 16],
                     max_contrl_epoch=165,
                     show_flag=True)

def lr_mod_4sgdr(iter):
    if iter >= len(lr_sgdr):
        return np.array(lr_sgdr).min()
    else:
        return lr_sgdr[iter]

# 注意,分类label必须从0开始标记
run(model = model,
    taskname = 'qwe',
    label_index = label_index,
    trainer = keras_trainer,
    tester = keras_tester4h5,
    predictor = keras_predictor,
    data_loader = data_loader,
    loss_func = loss_func,
    optimizer = optimizer,
    train_file_list = aug_train_list+or_train_list,
    test_file_list = or_test_list,
    ver_file_list = or_ver_list,
    ortrain_file_list = or_train_list,
    log_save_path = Result_save_Path,
    train_batchsize = train_batchsize,
    test_batchsize = test_batchsize,
    train_epoch = train_epoch,
    lr_plan = lr_mod_4sgdr,
    printer = color_printer,
    min_test_iter=10,
    test_iter_step=10,
    vis_model_save_iter_start=10,
    vis_model_save_iter_step=10,
    newest_model_save_iter_start=10,
    newest_model_save_iter_step=10)




