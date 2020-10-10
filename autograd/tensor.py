
from typing import List, Callable, Union

import numpy as np


class Dependency():

    def __init__(self, 
                 tensor:'Tensor',
                 grad_fn: Callable[[np.ndarray], np.ndarray]) -> None: 
        self.tensor = tensor
        self.grad_fn = grad_fn


Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


Tensorable = Union['Tensor', float, np.ndarray]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


# TODO: make a basic class of tensor, only with data and some func about data, then the "Tensor" class extends it.
class BasicTensor(object):
    def __init__(self, 
                 data: Arrayable) -> None:
        self.data = ensure_array(data)
        self.shape = self.data.shape
        self.ndim = self.data.ndim

    def __repr__(self) -> str:
        return f"Basic Tensor({self.data})"

    def tolist(self):
        return self.data.tolist()


# a tensor includes data and dependency info.
class Tensor(BasicTensor):

    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = []) -> None:
        # self.data = ensure_array(data)
        super().__init__(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on
        self.grad: np.ndarray = None

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    # t + other
    def __add__(self, other) -> 'Tensor':
        return t_add(self, other)

    # other + t
    def __radd__(self, other) :
        return t_add(other, self)
    
    # t += other
    #TODO
    def __iadd__(self, other):
        self.data = self.data + other.data
        return self

    # TODO: add mul, neg, sub, matmul and so on.

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data = self.grad.data + grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad)
            dependency.tensor.backward(backward_grad)


def t_add(t1:Tensor, t2:Tensor) -> Tensor:
    data: Tensor = t1.data + t2.data
    requires_grad: Tensor = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


if __name__ == "__main__":
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = Tensor([4, 5, 6], requires_grad=True)

    t3 = t1 + t2

    assert t3.data.tolist() == [5, 7, 9]

    t3.backward(Tensor([-1., -2., -3.]))

    assert t1.grad.tolist() == [-1, -2, -3]
    assert t2.grad.tolist() == [-1, -2, -3]