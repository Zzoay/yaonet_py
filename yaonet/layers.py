
from yaonet.module import Module
from yaonet.parameter import Parameter
from yaonet.tensor import Tensor


class Layer(Module):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super().__init__(input_shape, output_shape)

        self.w = Parameter(self.input_shape)
        self.b = Parameter()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.w + self.b
    
    def predict(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)