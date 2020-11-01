
import numpy as np

from yaonet.module import Module
from yaonet.tensor import Tensor
from yaonet.optimizers import SGD
from yaonet.layers import Layer, Linear
from yaonet.loss import MSE
from yaonet.basic_functions import sin, sigmoid

x_data = Tensor(np.random.randn(5000, 3))
coef1 = Tensor(np.array([[-1], [+3], [-2]], dtype=np.float))
coef2 = Tensor(np.array([2]))
y_data = sigmoid((x_data @ coef1) @ coef2) + 5
y_data = y_data.reshape(5000, 1)

epochs = 100
lr = 0.001
batch_size = 64
input_shape = 3
output_shape = 3


class Model(Layer):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super().__init__(input_shape, output_shape)
        self.linear1 = Linear(input_shape, output_shape)
        self.linear2 = Linear(output_shape, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear2(sigmoid(self.linear1(inputs)))

model = Model(input_shape, output_shape)
optimizer = SGD(module=model.parameters(for_optim=True), lr=lr)

loss_func = MSE()


if __name__ == "__main__":
    for epoch in range(epochs):
        epoch_loss = 0.0

        for start in range(0, x_data.shape[0], batch_size):
            end = start + batch_size

            optimizer.zero_grad()
            # model.zero_grad()  # as same

            x_batch = x_data[start:end]
            y_batch = y_data[start:end]

            y_pred = model(x_batch)
            loss = loss_func(y_pred, y_batch)

            loss.backward()
            epoch_loss += loss.data

            optimizer.step()

        print(f"Epoch {epoch}, loss: {epoch_loss},")

predicted = model(x_data)
print(loss_func(predicted, y_data).tolist() /5000)