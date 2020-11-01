
from typing import Union, Optional, Tuple

import numpy as np

from yaonet.tensor import Dependency, Tensor, ensure_tensor


def sin(t: Tensor) -> Tensor:
    t = ensure_tensor(t)
    
    data = np.sin(t.data)
    requires_grad = t.requires_grad

    depends_on = []

    if requires_grad: 
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # sin'(x) = cos(x)
            return grad * np.cos(t.data)

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, 
            requires_grad,
            depends_on)


def cos(t: Tensor) -> Tensor:
    t = ensure_tensor(t)
    
    data = np.cos(t.data)
    requires_grad = t.requires_grad

    depends_on = []

    if requires_grad: 
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # cos'(x) = -sin(x)
            return grad * (-np.sin(t.data))

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, 
            requires_grad,
            depends_on)


def tan(t: Tensor) -> Tensor:
    t = ensure_tensor(t)
    
    data = np.tan(t.data)
    requires_grad = t.requires_grad

    depends_on = []

    if requires_grad: 
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # tan'(x) = 1 / cos(x)**2
            return grad * (1 / np.power(np.cos(t.data),2))

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, 
            requires_grad,
            depends_on)


def power(t: Tensor, n: float) -> Tensor:
    t = ensure_tensor(t)

    data = np.power(t.data, n)
    requires_grad = t.requires_grad

    depends_on = []

    if requires_grad: 
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # pow(x, n) = n * pow(x, n-1)
            return grad * n * np.power(t.data, n-1)

        depends_on.append(Dependency(t, grad_fn))
    
    return Tensor(data, 
            requires_grad,
            depends_on)


def exp(t: Tensor) -> Tensor:
    t = ensure_tensor(t)

    data = np.exp(t.data)
    requires_grad = t.requires_grad

    depends_on = []

    if requires_grad: 
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # exp'(x) = exp(x)
            return grad * np.exp(t.data)

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, 
            requires_grad,
            depends_on)


def log(a: float, t: Tensor) -> Tensor:
    t = ensure_tensor(t)

    # np.log() is ln() default
    def loga_x(a, x):
        return np.log(x) / np.log(a)

    data = loga_x(a, t.data)
    requires_grad = t.requires_grad

    depends_on = []

    if requires_grad: 
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # loga'(x) = 1 / (x * ln(a)) 
            return grad * (1 / (t.data * np.log(a)))

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, 
            requires_grad,
            depends_on)


def sigmoid(t: Tensor) -> Tensor:
    t = ensure_tensor(t)  
    return 1 / (1 + exp(-t))


def tanh(t: Tensor) -> Tensor:
    t = ensure_tensor(t)
    return (exp(t) - exp(-t)) / (exp(t) + exp(-t))


def relu(t: Tensor) -> Tensor:
    t = ensure_tensor(t)

    data = np.maximum(0, t.data)
    requires_grad = t.requires_grad

    depends_on = []

    if requires_grad: 
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            new_grad = np.zeros_like(t.data)

            idxs = np.where(t.data > 0)
            new_grad[idxs] = grad[idxs] * t.data[idxs]
            
            return new_grad

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, 
            requires_grad,
            depends_on)


def max_pool1d(t: Tensor, 
               kernel_size: Union[Tensor, Tuple[Tensor, ...]], 
               stride: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None) -> Tensor:
    if stride is None:
        stride = kernel_size  # default

    # t shape: batch size, channel, L_in
    batch_size, channel, L_in = t.shape
    assert channel == 1  # simply for NLP tasks
    
    idx_tensor = np.zeros((batch_size, channel, L_in))
    for i in range(0, L_in, stride):
        tmp = t[:, :, i:i + stride]
        max_idx = np.argmax(tmp.data, axis=2).reshape(-1) + i 
        for c in range(channel):
            l = c*batch_size
            h = (c+1)*batch_size
            idx_tensor[list(range(batch_size)), c, max_idx.tolist()[l:h]] = 1

    result = t[idx_tensor.astype(bool)].reshape(batch_size, channel, -1)
    L_out = 1 + (L_in + - (kernel_size - 1) - 1 ) / stride
    assert result.shape[2] == L_out
    # result shape: batch size, channel, L_out

    return result
