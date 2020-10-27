
from typing import Generator, Callable

from yaonet.module import Module
from yaonet.parameter import Parameter


class Optimizer():
    def __init__(self, model_params: Parameter) -> None:
        self.model_params = model_params

    def zero_grad(self) -> None:
        for parameter in self.model_params():
            parameter.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, module: Module, lr: float = 0.01) -> None:
        super().__init__(module)
        self.lr = lr

    def step(self) -> None:
        for parameter in self.model_params():
            parameter -= self.lr * parameter.grad  # gradient descent
