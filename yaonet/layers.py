
import numpy as np

from typing import Union, Tuple

from yaonet.module import Module
from yaonet.parameter import Parameter
from yaonet.tensor import Tensor, Dependency


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
    def __init__(self, input_shape: int, output_shape: int, bias: bool = True) -> None:
        super().__init__(input_shape, output_shape)
        self.w = Parameter((self.input_shape, self.output_shape))
        self.bias = bias
        if self.bias:
            self.b = Parameter((1, self.output_shape))

    def forward(self, inputs: Tensor) -> Tensor:
        if self.bias:
            return inputs @ self.w + self.b
        return inputs @ self.w
    

class Conv2d(Layer):
    def __init__(self,
                in_channels: int, 
                out_channels: int, 
                kernel_size: Union[int, Tuple[int, int]], 
                stride: Union[int, Tuple[int, int]] = 1,
                bias: bool = True) -> None:          
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        self.stride = stride
        self.bias = bias

        self.w = Parameter((kernel_size[0], kernel_size[1], in_channels, out_channels))
        if self.bias:
            self.b = Parameter((1, out_channels))

    def forward(self, inputs: Tensor) -> Tensor:
        N, C, H, W  = inputs.shape
        out_h = (H - self.kernel_size[0]) // self.stride + 1
        out_w = (W - self.kernel_size[1]) // self.stride + 1

        self.col_weights = self.w.reshape([-1, self.out_channels])
        col = im2col(inputs, self.kernel_size, self.stride)
        if self.bias:
            output = col @ self.col_weights + self.b
        else:
            output = col @ self.col_weights
        return output.reshape([N, self.out_channels, out_h, out_w])  # B, C, OH, OW
    

def im2col(image: Tensor, ksize: Union[Tensor, Tuple[Tensor, Tensor]], stride: Union[Tensor, Tuple[Tensor, Tensor]] = 1):
    batchsize, channel, height, width= image.shape

    ksize1 = ksize
    ksize2 = ksize
    if isinstance(ksize, (tuple,list)) and len(ksize) == 2:    
        ksize1, ksize2 = ksize

    image_col = []
    for i in range(0, height - ksize1 + 1, stride):
        for j in range(0, width - ksize2 + 1, stride):
            col = image[:, :, i:i + ksize1, j:j + ksize2].data.reshape(batchsize, channel, -1)
            image_col.append(col)
    image_col = np.array(image_col)
    image_col = image_col.reshape(*list(image_col.shape)[:2], -1).transpose(1,0,2)

    requires_grad = image.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            new_grad = np.zeros((batchsize, channel, height, width))
            
            cnt = 0
            for i in range(0, height - ksize1 + 1, stride):
                for j in range(0, width - ksize2 + 1, stride):
                    new_grad[:, :, i:i + ksize1, j:j + ksize2] += grad[:, cnt, :].reshape(batchsize, channel, ksize1, ksize2)
                    cnt += 1
            return new_grad

        depends_on = [Dependency(image, grad_fn)]
    else:
        depends_on = []

    return Tensor(image_col,
                  requires_grad,
                  depends_on)


class Embedding(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.num_embed = num_embeddings  # usually vocab_size
        self.embed_dim = embedding_dim
        self.weights = Parameter((num_embeddings, embedding_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        #  inputs.shape: batch_size, sentence_length

        emb = []  # batch_size, sentence_length, embedding_dim
        for item in inputs:
            emb.append(self.weights[item.tolist()].toarray())

        requires_grad = True
        depends_on = []
        if requires_grad:
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                assert grad.shape == np.shape(emb)

                if self.weights.grad is None:
                    self.weights.zero_grad()
                pre_grad = self.weights.grad
                for i,item in enumerate(inputs):
                    pre_grad[item.tolist()] += grad[i, :, :]
                return pre_grad

            depends_on = [Dependency(self.weights, grad_fn)]
        else:
            depends_on = []

        return Tensor(emb,
                      requires_grad,
                      depends_on)  


class RNN(Layer):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super().__init__(input_shape, output_shape)

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError


class LSTM(Layer):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super().__init__(input_shape, output_shape)

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError


class GRU(Layer):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super().__init__(input_shape, output_shape)

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError