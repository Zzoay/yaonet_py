
import numpy as np

from yaonet.module import Module
from yaonet.tensor import Tensor
from yaonet.optim import SGD
from yaonet.layers import Layer, Linear, Conv2d, Embedding
from yaonet.functional import sigmoid, relu, max_pool1d, mean_squared_error
from yaonet.utils import clip_grad_value


x_data = Tensor(np.random.randint(500, size=12000).reshape(2000, -1))
coef1 = Tensor(np.array([[-1], [+3], [-2], [1], [+4], [2]], dtype=np.float))
coef2 = Tensor(np.array([2]))
y_data = (x_data/500 @ coef1) @ coef2 + 5
y_data = y_data.reshape(2000, 1)

epochs = 20
lr = 0.001
batch_size = 64
input_shape = 3
output_shape = 3

embed_dim = 30
kernel_num = 2
kernel_sizes =  [3]
stride = 1


class Model(Layer):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        self.emb = Embedding(500, embed_dim)
        self.conv = Conv2d(in_channels=1, out_channels=kernel_num, kernel_size=(kernel_sizes[0], embed_dim), stride=stride)
        self.linear1 = Linear(kernel_num*len(kernel_sizes), output_shape)
        self.linear2 = Linear(output_shape, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        emb_x = self.emb(inputs)
        batch_size, sentence_length, embed_dim = emb_x.shape
        emb_x = emb_x.reshape(batch_size, 1, sentence_length, embed_dim)
        
        conv_output = self.conv(emb_x)
        h1 = relu(conv_output.squeeze(3))
        h2 = max_pool1d(h1, h1.shape[2]).squeeze(2)
        h3 = sigmoid(self.linear1(h2))
        logits = self.linear2(h3)
        return logits


model = Model(input_shape, output_shape)
optimizer = SGD(module=model.parameters(for_optim=True), lr=lr)


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
            loss = mean_squared_error(y_pred, y_batch)

            loss.backward()
            epoch_loss += loss.data

            clip_grad_value(model.parameters(), clip_value=10)
            optimizer.step()

        print(f"Epoch {epoch}, loss: {epoch_loss},")

predicted = model(x_data)
# print(*zip(predicted.tolist(), y_data.tolist()))
rmse = np.sqrt((predicted - y_data).data**2).mean()
print(rmse)