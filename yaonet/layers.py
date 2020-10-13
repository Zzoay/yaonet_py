
from yaonet.module import Module
from yaonet.parameter import Parameter
from yaonet.tensor import Tensor


class Layer(Module):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def predict(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)


class Linear(Layer):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super().__init__(input_shape, output_shape)
        self.w = Parameter(self.input_shape, self.output_shape)
        self.b = Parameter(1, self.output_shape)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.w + self.b
    

class CNN(Layer):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super().__init__(input_shape, output_shape)

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError