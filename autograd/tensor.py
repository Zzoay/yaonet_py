
import numpy as np

from typing import List, Callable, Union


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


Tensorable = Union['BasicTensor', 'Tensor', float, np.ndarray]

def ensure_basicTensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, BasicTensor):
        return tensorable
    else:
        return BasicTensor(tensorable)


def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


# make a basic class of tensor, only with data and some func about data, then the "Tensor" class extends it.
class BasicTensor(object):
    def __init__(self, 
                 data: Arrayable) -> None:
        self._data = ensure_array(data)
        self.shape = self.data.shape
        self.ndim = self._data.ndim
      
    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data

    def __repr__(self) -> str:
        return f"Basic Tensor({self.data})"

    def tolist(self):
        return self.data.tolist()

    def sum(self, **kwargs):
        return ensure_basicTensor(self.data.sum(**kwargs))

    def __add__(self, other):
        return ensure_basicTensor(self.data + ensure_basicTensor(other).data)
    
    def __radd__(self, other):
        return ensure_basicTensor(ensure_basicTensor(other).data + self.data)

    # t += other
    def __iadd__(self, other):
        self.data = self.data + ensure_tensor(other).data
        return self


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
        self.grad: BasicTensor = None

        if self.requires_grad:
            self.zero_grad()
      
    @property
    def data(self) -> np.ndarray:
        return super().data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        super(Tensor, Tensor).data.__set__(self, new_data)
        # set a tensor means that its grad should become None.
        self.grad = None

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        self.grad = BasicTensor(np.zeros_like(self.data, dtype=np.float64))

    def sum(self) -> 'Tensor':
        return t_sum(self)

    # t + other
    def __add__(self, other) -> 'Tensor':
        return t_add(self, ensure_tensor(other))

    # other + t
    def __radd__(self, other) :
        return t_add(ensure_tensor(other), self)
    
    # def __iadd__(self, other):
    #     self.data = self.data + ensure_tensor(other).data
    #     return self

    # TODO: add mul, neg, sub, matmul and so on.
    def __mul__(self, other):
        raise NotImplementedError

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = BasicTensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad = self.grad + grad

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad)
            dependency.tensor.backward(backward_grad)


def t_sum(t:Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements.
    """
    data: Tensor = t.data.sum()
    requires_grad: bool = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor, so each input element
            contributes that much
            """
            return grad * ensure_basicTensor(np.ones_like(t.data))

        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def t_add(t1:Tensor, t2:Tensor) -> Tensor:
    data: Tensor = t1.data + t2.data
    requires_grad: bool = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t1.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return ensure_basicTensor(grad)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t2.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return ensure_basicTensor(grad)

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def t_mul(t1:Tensor, t2:Tensor) -> Tensor:
    data: Tensor = t1.data * t2.data
    requires_grad: bool = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    return Tensor(
            data,
            requires_grad,
            depends_on)

