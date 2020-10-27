
import numpy as np

from yaonet.module import Module
from yaonet.tensor import Tensor
from yaonet.optimizers import SGD
from yaonet.layers import Layer, Linear, Conv2d, Embedding
from yaonet.loss import MSE
from yaonet.functions import sin, sigmoid

x_data = Tensor(np.random.randint(500, size=6000).reshape(2000, -1))
coef1 = Tensor(np.array([[-1], [+3], [-2]], dtype=np.float))
coef2 = Tensor(np.array([2]))
y_data = sigmoid((x_data @ coef1) @ coef2) + 5
y_data = y_data.reshape(2000, 1)

epochs = 30
lr = 0.001
batch_size = 64
input_shape = 3
output_shape = 3

embed_dim = 30
ksize = 3
stride = 1


class Model(Layer):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        self.emb = Embedding(500, embed_dim)
        self.linear1 = Linear(30, 1)
        self.linear2 = Linear(output_shape, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        emb_x = self.emb(inputs)
        batch_size, sentence_length, embed_dim = emb_x.shape
        
        h1 = self.linear1(emb_x)
        h1 = h1.reshape(batch_size, -1)
        h2 = self.linear2(sigmoid(h1))
        return h2

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
# print(*zip(predicted.tolist(), y_data.tolist()))
print(loss_func(predicted, y_data).tolist() /2000)