import numpy as np
from src.layers.activation import ActivationLayer


class TanhLayer(ActivationLayer):
    def __init__(self) -> None:
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)
