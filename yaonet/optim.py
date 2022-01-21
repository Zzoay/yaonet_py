
from typing import Generator, Callable

from yaonet.module import Module
from yaonet.parameter import Parameter


class Optimizer():
    def __init__(self, params: Callable) -> None:
        self.params = params  # actually, model_params is a Callable type 

    def zero_grad(self) -> None:
        for parameter in self.params():  # type:ignore
            parameter.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: Callable, lr: float = 0.01) -> None:
        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        for parameter in self.params():
            parameter -= self.lr * parameter.grad  # gradient descent
