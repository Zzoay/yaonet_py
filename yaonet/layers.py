
import numpy as np

from typing import Union, Tuple, Optional

from yaonet.module import Module
from yaonet.parameter import Parameter
from yaonet.tensor import Tensor, Dependency
from yaonet.functional import sigmoid, tanh
from yaonet.utils import cat


class Layer(Module):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs: Tensor):
        raise NotImplementedError

    def predict(self, inputs: Tensor):
        return self.forward(inputs)

    def __call__(self, inputs: Tensor):
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
                stride: int = 1,  # TODO: stride could be a tuple
                bias: bool = True) -> None:          
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size          
        self.stride = stride
        self.bias = bias

        self.w = Parameter((self.kernel_size[0], self.kernel_size[1], in_channels, out_channels))
        if self.bias:
            self.b = Parameter((1, out_channels))

    def forward(self, inputs: Tensor) -> Tensor:
        N, _, H, W  = inputs.shape   # batch size, input channel, input height, input width
        out_h = (H - self.kernel_size[0]) // self.stride + 1
        out_w = (W - self.kernel_size[1]) // self.stride + 1

        self.col_weights = self.w.reshape([-1, self.out_channels])
        col = im2col(inputs, self.kernel_size, self.stride)
        if self.bias:
            output = col @ self.col_weights + self.b
        else:
            output = col @ self.col_weights
        return output.reshape([N, self.out_channels, out_h, out_w])  # batch size, output channel, output height, output width
    

def im2col(image: Tensor, ksize: Tuple[int, int], stride: int = 1):
    batchsize, channel, height, width= image.shape

    # ksize1 = ksize
    # ksize2 = ksize
    if isinstance(ksize, (tuple,list)) and len(ksize) == 2:    
        ksize1, ksize2 = ksize[0], ksize[1]

    image_col = []
    for i in range(0, height - ksize1 + 1, stride):
        for j in range(0, width - ksize2 + 1, stride):
            col = image[:, :, i:i + ksize1, j:j + ksize2].data.reshape(batchsize, channel, -1)
            image_col.append(col)
    image_col = np.array(image_col)  # type: ignore
    image_col = image_col.reshape(*list(image_col.shape)[:2], -1).transpose(1,0,2)  # type: ignore

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
        for item in inputs[:]:
            emb.append(self.weights[item.tolist()].toarray())

        requires_grad = True
        depends_on = []
        if requires_grad:
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                assert grad.shape == np.shape(emb)

                if self.weights.grad is None:
                    self.weights.zero_grad()
                pre_grad = self.weights.grad
                for i,item in enumerate(inputs[:]):
                    pre_grad[item.tolist()] += grad[i, :, :]  # type: ignore
                return pre_grad   # type: ignore

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

# previous version
class LSTM_(Layer):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 num_layers: int = 1, 
                 bias: bool = True) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        
        self.i_input_weights = Parameter((input_size, hidden_size))
        self.i_forget_weights = Parameter((input_size, hidden_size))
        self.i_cell_weights = Parameter((input_size, hidden_size))
        self.i_output_weights = Parameter((input_size, hidden_size))

        self.h_input_weights = Parameter((hidden_size, hidden_size))
        self.h_forget_weights = Parameter((hidden_size, hidden_size))
        self.h_cell_weights = Parameter((hidden_size, hidden_size))
        self.h_output_weights = Parameter((hidden_size, hidden_size))

        assert bias, "In nlp tasks, bias usually exists"
        self.i_input_bias = Parameter((1, hidden_size))
        self.i_forget_bias = Parameter((1, hidden_size))
        self.i_cell_bias = Parameter((1, hidden_size))
        self.i_output_bias = Parameter((1, hidden_size))

        self.h_input_bias = Parameter((1, hidden_size))
        self.h_forget_bias = Parameter((1, hidden_size))
        self.h_cell_bias = Parameter((1, hidden_size))
        self.h_output_bias = Parameter((1, hidden_size))

    def forward(self, inputs: Tensor, hc_0: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # default batch first
        # input shape: (batch_size, seq_length, input_size)
        batch_size, seq_length, input_size = inputs.shape
        if hc_0:
            h_t, c_t = hc_0
        else:
            h_t = Tensor(np.zeros((batch_size, self.num_layers, self.hidden_size)))
            c_t = Tensor(np.zeros((batch_size, self.num_layers, self.hidden_size)))

        h_lst = []
        for i in range(seq_length):
            x_t = inputs[:, i, :].unsqueeze(1)
            
            i_t = sigmoid(x_t @ self.i_input_weights + self.i_input_bias + h_t @ self.h_input_weights + self.h_input_bias)
            f_t = sigmoid(x_t @ self.i_forget_weights + self.i_forget_bias + h_t @ self.h_forget_weights + self.h_forget_bias)
            g_t = tanh(x_t @ self.i_cell_weights + self.i_cell_bias + h_t @ self.h_cell_weights + self.h_cell_bias)
            o_t = sigmoid(x_t @ self.i_output_weights + self.i_output_bias + h_t @ self.h_output_weights + self.h_output_bias)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * tanh(c_t)
            
            h_lst.append(h_t)
        
        outputs = cat(h_lst, axis=1)

        # output_shape: (batch_size, seq_length, hidden_size)
        assert outputs.shape == (batch_size, seq_length, self.hidden_size)
        assert h_t.shape == (batch_size, self.num_layers, self.hidden_size)
        assert c_t.shape == (batch_size, self.num_layers, self.hidden_size)

        return outputs, (h_t, c_t)


class LSTM(Layer):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 num_layers: int = 1, 
                 bias: bool = True) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.input_weights = Parameter((input_size + hidden_size, hidden_size))
        self.forget_weights = Parameter((input_size + hidden_size, hidden_size))
        self.cell_weights = Parameter((input_size + hidden_size, hidden_size))
        self.output_weights = Parameter((input_size + hidden_size, hidden_size))

        if bias:
            self.input_bias = Parameter((1, hidden_size))
            self.forget_bias = Parameter((1, hidden_size))
            self.cell_bias = Parameter((1, hidden_size))
            self.output_bias = Parameter((1, hidden_size))

    def forward(self, inputs: Tensor, hc_0: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # default batch first
        # input shape: (batch_size, seq_length, input_size)
        batch_size, seq_length, _input_size = inputs.shape
        # h_t is the hidden state at time t, c_t is the cell state at time t; if (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        if hc_0:
            h_t, c_t = hc_0
        else:
            h_t = Tensor(np.zeros((batch_size, self.num_layers, self.hidden_size)))
            c_t = Tensor(np.zeros((batch_size, self.num_layers, self.hidden_size)))

        h_lst = []
        for i in range(seq_length):
            x_t = inputs[:, i, :].unsqueeze(1)
            xh_t = cat((x_t, h_t), axis=2)

            if self.bias:
                # i_t, f_t, g_t, o_t are the input, forget, cell, and output gates
                i_t = sigmoid(xh_t @ self.input_weights + self.input_bias)
                f_t = sigmoid(xh_t @ self.forget_weights + self.forget_bias)
                g_t = tanh(xh_t @ self.cell_weights + self.cell_bias)
                o_t = sigmoid(xh_t @ self.output_weights + self.output_bias)
            else:
                i_t = sigmoid(xh_t @ self.input_weights)
                f_t = sigmoid(xh_t @ self.forget_weights)
                g_t = tanh(xh_t @ self.cell_weights)
                o_t = sigmoid(xh_t @ self.output_weights)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * tanh(c_t)
            
            h_lst.append(h_t)
        
        outputs = cat(h_lst, axis=1)
        
        # output_shape: (batch_size, seq_length, hidden_size)
        assert outputs.shape == (batch_size, seq_length, self.hidden_size)
        assert h_t.shape == (batch_size, self.num_layers, self.hidden_size)
        assert c_t.shape == (batch_size, self.num_layers, self.hidden_size)

        return outputs, (h_t, c_t)


class GRU(Layer):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super().__init__(input_shape, output_shape)

    def forward(self, inputs: Tensor) -> Tensor:  
        raise NotImplementedError