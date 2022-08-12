import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from src.layers.convolutional import ConvolutionalLayer
from src.layers.dense import DenseLayer
from src.layers.reshape import ReshapeLayer
from src.layers.sigmoid import SigmoidLayer
from src.layers.softmax import SoftmaxLayer
from src.loss import binary_cross_entropy, binary_cross_entropy_prime
from src.network import process, train


def preprocess_data(x, y, limit):
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x[:limit], y[:limit]


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

network = [
    ConvolutionalLayer((1, 28, 28), 3, 5),
    SigmoidLayer(),
    ReshapeLayer((5, 26, 26), (5 * 26 * 26, 1)),
    DenseLayer(5 * 26 * 26, 100),
    SigmoidLayer(),
    DenseLayer(100, 100),
    SigmoidLayer(),
    DenseLayer(100, 10),
    SoftmaxLayer(),
]

train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=1000,
    learning_rate=0.1,
)


ok = 0
for x, y in zip(x_test, y_test):
    output = process(network, x)
    if np.argmax(output) == np.argmax(y):
        ok += 1
    print(f"{np.argmax(output)} excepted {np.argmax(y)}")

print(f"{ok}/{len(x_test)}")
