import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


def autoencoder1():
    # 加载MNIST数据集
    (x_train, _), (x_test, _) = mnist.load_data()

    # 归一化和预处理
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # 设置编码维度
    encoding_dim = 32

    # 输入层
    input_img = Input(shape=(784,))

    # 编码层
    encoded = Dense(encoding_dim, activation='relu')(input_img)

    # 解码层
    decoded = Dense(784, activation='sigmoid')(encoded)

    # 构建自编码器模型
    autoencoder = Model(input_img, decoded)

    # 编码模型
    encoder = Model(input_img, encoded)

    # 为解码器创建输入
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # 编译和训练自编码器
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # 编码和解码一些手写数字
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    # 可视化原始和重建图像
    n = 10  # 展示10个数字
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # 原始图像
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 重建图像
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()





if __name__ == '__main__':
    autoencoder1()