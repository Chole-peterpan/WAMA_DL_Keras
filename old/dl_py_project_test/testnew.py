from keras.models import *
from keras.layers import Input, Conv2D, Dense, BatchNormalization, concatenate
from keras.layers.core import Flatten
import matplotlib.pyplot as plt

def testnet():
    inputs_1 = Input(shape=(13, 11, 1), name='input1')
    x1 = Conv2D(100, (3, 3), name='conv1', use_bias=False, kernel_initializer='Zeros')(inputs_1)
    x1 = BatchNormalization(axis=1, epsilon=1e-6, trainable=True, name='bn1')(x1)
    x2 = BatchNormalization(axis=2, epsilon=1e-6, trainable=True, name='bn2')(x1)
    x3 = BatchNormalization(axis=3, epsilon=1e-6, trainable=True, name='bn3')(x1)

    x1 = Flatten()(x1)
    x2 = Flatten()(x2)
    x3 = Flatten()(x3)

    x=concatenate([x1, x2, x3], axis=-1)
    out = Dense(2, name='dense1',use_bias=False, kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs_1, outputs=out, name='dual input net')
    return model

model = testnet()
print(model.summary())




# 由summary可以看出，bn层的参数数量等于对应轴的神经元数量（或特征图数量）*4
# 4可能代表了4个可更新的参数：
BN1 = model.get_layer(name='bn1')
BN1_weight = BN1.get_weights()[0]

conv1_weight = np.reshape(conv1_weight, conv1_weight.size)
conv1_bias = conv1.get_weights()[1]
plt.subplot(1, 2, 1)
plt.hist(conv1_weight, bins=200, range=[conv1_weight.min()-0.2, conv1_weight.max()+0.2], color='red')
plt.title('he_uniform4weight')
plt.subplot(1, 2, 2)
plt.hist(conv1_bias, bins=200, range=[conv1_bias.min()-0.2, conv1_bias.max()+0.2])
plt.title('RandomNormal4bias')
plt.show()









