from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# step1: import model finished
from keras.applications.vgg16 import VGG16
import numpy as np
from vis.visualization import visualize_cam
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Conv3D, MaxPooling3D, ZeroPadding2D, merge,Activation
from keras.optimizers import SGD
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image    #image：用于图像数据的实时数据增强的（相当）基本的工具集。
import numpy as np
import matplotlib.pyplot as plt
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam, visualize_activation, visualize_cam_with_losses
from keras import activations
import matplotlib.cm as cm
import cv2
from function import pause
import h5py
# from __future__ import absolute_import
from vis.input_modifiers import Jitter
import numpy as np
from scipy.ndimage.interpolation import zoom

from keras.layers.convolutional import _Conv
from keras.layers.pooling import _Pooling1D, _Pooling2D, _Pooling3D
from keras.layers.wrappers import Wrapper
from keras import backend as K


# from vis import utils
from vis.losses import ActivationMaximization
from vis.backprop_modifiers import get
from vis.optimizer import Optimizer
from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# step1: import model finished
from keras.applications.vgg16 import VGG16
import numpy as np
from vis.visualization import visualize_cam
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Conv3D, MaxPooling3D, ZeroPadding2D, merge,Activation
from keras.optimizers import SGD
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image    #image：用于图像数据的实时数据增强的（相当）基本的工具集。
import numpy as np
import matplotlib.pyplot as plt
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam, visualize_activation, visualize_cam_with_losses
from keras import activations
import matplotlib.cm as cm
import cv2
import h5py
# from __future__ import absolute_import

import numpy as np
from scipy.ndimage.interpolation import zoom

from keras.layers.convolutional import _Conv
from keras.layers.pooling import _Pooling1D, _Pooling2D, _Pooling3D
from keras.layers.wrappers import Wrapper
from keras import backend as K


# from vis import utils
from vis.losses import ActivationMaximization
from vis.backprop_modifiers import get
from vis.optimizer import Optimizer
from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# step1: import model finished
from function import pause, test_on_model4_subject, test_on_model4_subject4_or_train
from w_dualpathnet import dual_path_net
from w_resnet import resnet, resnet_nobn, se_resnet
from w_dense_net import dense_net,se_dense_net
from w_resnext import  resnext, resnext_or, se_resnext
from w_vggnet import vgg16_w_3d, vgg16_w_3d_gb
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss
from keras.applications.vgg16 import VGG16
import numpy as np
from vis.visualization import visualize_cam
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Conv3D, MaxPooling3D, ZeroPadding2D, merge,Activation
from keras.optimizers import SGD
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image    #image：用于图像数据的实时数据增强的（相当）基本的工具集。
import numpy as np
import matplotlib.pyplot as plt
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam, visualize_activation, visualize_cam_with_losses
from keras import activations
import matplotlib.cm as cm
import cv2
import h5py
# from __future__ import absolute_import

import numpy as np
from scipy.ndimage.interpolation import zoom

from keras.layers.convolutional import _Conv
from keras.layers.pooling import _Pooling1D, _Pooling2D, _Pooling3D
from keras.layers.wrappers import Wrapper
from keras import backend as K


# from vis import utils
from vis.losses import ActivationMaximization
from vis.backprop_modifiers import get
from vis.optimizer import Optimizer
from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# step1: import model finished
from function import pause, test_on_model4_subject, test_on_model4_subject4_or_train
from w_dualpathnet import dual_path_net
from w_resnet import resnet, resnet_nobn, se_resnet
from w_dense_net import dense_net,se_dense_net
from w_resnext import  resnext, resnext_or, se_resnext
from w_vggnet import vgg16_w_3d, vgg16_w_3d_gb
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss
from keras.applications.vgg16 import VGG16
import numpy as np
from vis.visualization import visualize_cam
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Conv3D, MaxPooling3D, ZeroPadding2D, merge,Activation
from keras.optimizers import SGD
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from w_resnet import resnet
from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import random
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# step1: import model finished
from function import pause, test_on_model4_subject, test_on_model4_subject4_or_train, lr_mod
from w_dualpathnet import dual_path_net
from w_resnet import resnet, resnet_nobn, se_resnet
from w_dense_net import dense_net,se_dense_net
from w_resnext import  resnext, resnext_or, se_resnext
from w_vggnet import vgg16_w_3d, vgg16_w_3d_gb
from w_loss import  EuiLoss, y_t, y_pre, Acc, EuclideanLoss

import keras.backend as K
from w_resnet import resnet_nobn, se_resnet, resnet_or
import keras.backend as K

# step2: import extra model finished




# step2: import extra model finished


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))



# 構建網絡
model = resnet_or(use_bias_flag=True,classes=2)
# model.compile(optimizer=SGD(lr=1e-6, momentum=0.9), loss='binary_crossentropy')
# 構建輸入
data_input_c = np.zeros([1, 280, 280, 16, 1], dtype=np.float32)
H5_file = h5py.File(r'/data/@data_liaoxiao/4test/22_1.h5', 'r')
batch_x = H5_file['data'][:]
batch_y = H5_file['label3'][:]
H5_file.close()
batch_x = np.transpose(batch_x, (1, 2, 0))
data_input_c[0, :, :, :, 0] = batch_x[:, :, :]


# 保存或加載權重
# pre = model.predict_on_batch(data_input_c)
# model.save('G:\qweqweqweqwe\model.h5')
model.load_weights(filepath='/data/XS_Aug_model_result/model_templete/diploma_real/@test40/m_40000_model.h5',by_name=True)


# 查看計算圖
print(model.summary())

# 查詢要做Activation Maximization的層的index
# 這裡我們可視化兩個層，一個是最後一個全連接層以觀察激活某一類的理想input，另一個是一個卷積層以觀察這個卷積層各個卷積核的特征
# layer_idx_fc = utils.find_layer_idx(model, 'fc1')
# layer_idx_conv = utils.find_layer_idx(model, 'L1_block1conv2')


# # 首先按照套路先把全連接層的激活函數改為線性激活
# model.layers[layer_idx_fc].activation = activations.linear
# model = utils.apply_modifications(model)

#
# # 開始全連接層的可視化（普通版本）==================================================================================
# # 注意參數seed_input，這個相當於input，也就是初始的input，這個input經過不斷的優化（梯度下降）逐漸實現最大激活
# # 一般情況下，全連接層的seed_input無需指定，函數內部默認生成隨機值作為seed_input
# # 當生成的最大激活input效果不好時，你可以把這個不好的結果作為seed_input進行下一次輸入，
# # 之後不斷迭代，即可改善結果的觀感（不過這樣貌似和增加步長作用差不多？）
# max_activition_norm = visualize_activation(model, layer_idx_fc, filter_indices=1,  verbose=True)
# max_activition_norm = np.squeeze(max_activition_norm, axis=-1)
# for ii in range(16):
#     plt.figure()
#     max_activition_piece = max_activition_norm[:, :, ii]
#     plt.imshow(max_activition_piece, cmap=plt.cm.gray)
#     plt.show()
#
# # 接下來開始逐漸改善最大激活圖
# # 法一：增加優化的最大步長
# max_activition_better = visualize_activation(model, layer_idx_fc, filter_indices=1, max_iter=500,  verbose=True)
# max_activition_better = np.squeeze(max_activition_better, axis=-1)
# for ii in range(16):
#     plt.figure()
#     max_activition_piece = max_activition_better[:, :, ii]
#     plt.imshow(max_activition_piece, cmap=plt.cm.jet)
#     plt.show()
#
#
# # 法二：在優化過程中使輸入抖動（即最後返回的最大激活圖），每次抖動16個像素值
# # 另外Jitter的參數如果大於1則代表像素個數（或體素個數），如果小於1大於0則代表圖像size的比例（乘以長和寬)
# max_activition_better = visualize_activation(model, layer_idx_fc, filter_indices=1, max_iter=5000, input_modifiers=[Jitter(8)], verbose=True)
# max_activition_better = np.squeeze(max_activition_better, axis=-1)
# for ii in range(16):
#     plt.figure()
#     max_activition_piece = max_activition_better[:, :, ii]
#     plt.imshow(max_activition_piece, cmap=plt.cm.jet)
#     plt.show()
#
#
# # 我們還可以指定多各類，這樣最大激活input的觀感就有點像“四不像”，即多各類的物體組成的某種東西，很有趣
# max_activition_better = visualize_activation(model, layer_idx_fc, filter_indices=[0,1], max_iter=500, input_modifiers=[Jitter(16)], verbose=True)
#
#



layer_idx_conv = utils.find_layer_idx(model, 'predictions')
model.layers[layer_idx_conv].activation = activations.linear
model = utils.apply_modifications(model)
filter_indices =392
# 卷積層的卷積核的最大激活input===========================================================================
# 觀察第一個卷積層的第2個卷積核(filter_indices=1)的最大激活input，這個最大激活input即可表示這個卷積核所關注並提取的特征（人為可理解的）
# data_input_c[:,:,:,0]=max_activition_norm
# max_activition_norm = visualize_activation(model, layer_idx_conv, filter_indices=filter_indices, max_iter=1000, input_modifiers=[Jitter(9)],  verbose=True,seed_input=data_input_c)
max_activition_norm = visualize_activation(model, layer_idx_conv, filter_indices=filter_indices, max_iter=5000, input_modifiers=[Jitter(16)],  verbose=True)
# max_activition_norm = np.squeeze(max_activition_norm, axis=-1)
for ii in range(1):
    plt.figure()
    # max_activition_piece = max_activition_norm[:, :, ii]
    plt.imshow(max_activition_norm)
    # plt.imshow((data_input_c[:,:,ii,0]), cmap=plt.cm.jet)
    plt.title(str(filter_indices))
    plt.show()
    # plt.figure()
    # plt.imshow(max_activition_norm)
    # # plt.imshow((data_input_c[:,:,ii,0]), cmap=plt.cm.jet)
    # plt.title(str(filter_indices))
    # plt.show()

# pause()
# 改善的方法同上，另外還可以另tv權重為0（暫時沒搞懂這個參數），或者指定不同的 regularization weights.
# max_activition_norm = visualize_activation(model, layer_idx_conv, filter_indices=1,  verbose=True, tv_weight=0.)


