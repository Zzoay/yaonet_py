
import numpy as np

import time

from yaonet.layers import LSTM, Linear, Layer
from yaonet.tensor import Tensor
from yaonet.functional import sigmoid, relu, max_pool1d, mean_squared_error
from yaonet.optim import SGD
from yaonet.utils import clip_grad_value

s_time = time.time()

x_data = Tensor(np.random.rand(2000, 3, 10))
coef = Tensor(np.random.rand(10, 1))
y_data = (x_data @ coef).squeeze(-1) @ Tensor(np.array([3, -2, -5]))
y_data = y_data.reshape(-1, 1)

epochs = 20
batch_size = 64
lr = 0.001


class Model(Layer):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        self.lstm = LSTM(input_size=10, hidden_size=10)
        self.linear = Linear(10, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        lstm_output, (_, _) = self.lstm(inputs)
        batch_size, sentence_length, embed_dim = inputs.shape

        h1 = relu(lstm_output).reshape(batch_size, -1, 3)
        h2 = max_pool1d(h1, h1.shape[2]).squeeze(2)
        logits = self.linear(h2)
        return logits


model = Model(10, 10)
optimizer = SGD(module=model.parameters(for_optim=True), lr=lr)


if __name__ == "__main__":
    step = 0
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

            if step % 10 == 0:
                print(f"Step: {step}, loss: {loss}")

            loss.backward()
            epoch_loss += loss.data

            clip_grad_value(model.parameters(), clip_value=10)
            optimizer.step()

            step += 1

        print(f"Epoch {epoch}, loss: {epoch_loss},")

predicted = model(x_data)
# print(*zip(predicted.tolist(), y_data.tolist()))
rmse = np.sqrt((predicted - y_data).data**2).mean()
print(rmse)

print(f"Time cost: {time.time()-s_time} s")
# Epoch 19, loss: 10959.21770824176,
# RMSE: 1.8802128625332934
# Time cost: 4218.714826107025 s