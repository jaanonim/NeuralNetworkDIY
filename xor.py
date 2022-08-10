import numpy as np

from src.layers.dense import DenseLayer
from src.layers.tanh import TanhLayer
from src.loss import mse, mse_prime
from src.network import process, train


def main():
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

    network = [DenseLayer(2, 3), TanhLayer(), DenseLayer(3, 1), TanhLayer()]

    train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.2)

    for x, y in zip(X, Y):
        out = process(network, x)
        print(f"{x[0]} -> {out[0]} excepted {y[0]}")


if __name__ == "__main__":
    main()
