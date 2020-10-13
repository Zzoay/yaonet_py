
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

    def toarray(self):
        return self.data


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
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def sum(self) -> 'Tensor':
        return t_sum(self)

    # t + other
    def __add__(self, other) -> 'Tensor':
        return t_add(self, ensure_tensor(other))

    # other + t
    def __radd__(self, other) :
        return t_add(ensure_tensor(other), self)
    
    # t = t + other
    def __iadd__(self, other):
        self.data = self.data + ensure_tensor(other).data
        return self

    def __mul__(self, other):
        return t_mul(self, ensure_tensor(other))
    
    def __rmul__(self, other) :
        return t_mul(ensure_tensor(other), self)
    
    def __imul__(self, other):
        self.data = self.data * ensure_tensor(other).data
        return self

    def __neg__(self):
        return t_neg(self)

    def __sub__(self, other):
        return t_sub(self, ensure_tensor(other))
    
    def __rsub__(self, other):
        return t_sub(ensure_tensor(other), self)

    def __isub__(self, other):
        self.data = self.data - ensure_tensor(other).data
        return self

    def __matmul__(self, other):
        return t_matmul(self, ensure_tensor(other))

    def __getitem__(self, index):
        return _slice(self, index)
        
    def backward(self, grad: np.ndarray = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = np.array(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")
        elif isinstance(grad, Tensor):
            grad = grad.toarray()
        elif isinstance(grad, List):
            grad = np.array(grad)

        self.grad = self.grad + grad

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad)
            dependency.tensor.backward(backward_grad)


def t_sum(t: Tensor) -> Tensor:
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
            # the gradient of function 'sum' is 1 to all elements
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def t_add(t1: Tensor, t2: Tensor) -> Tensor:
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

            return grad

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

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def t_mul(t1: Tensor, t2: Tensor) -> Tensor:
    data: Tensor = t1.data * t2.data
    requires_grad: bool = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # chain rule
            grad = grad * t2.data

            # Sum out added dims
            ndims_added = grad.ndim - t1.ndim
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
            # chain rule
            grad = grad * t1.data

            # Sum out added dims
            ndims_added = grad.ndim - t2.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(
            data,
            requires_grad,
            depends_on)


def t_neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad

    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def t_sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2


def t_matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.matmul(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # where t3 = t1 @ t2 (@ means matmul), t1 is (a ,b), t2 is (b, c) -> t3 (a, c)
            # chain rule: grad1 = grad3 @ t2.T
            return np.matmul(grad, t2.data.T)  # 'grad @ t2.data.T' also work

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # chain rule: grad2 = t1.T @ grad3 
            return np.matmul(t1.data.T, grad)

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _slice(t: Tensor, idxs) -> Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # only the idxs place can recieve grad, other is 0
            whole_t_grad = np.zeros_like(t.data)
            whole_t_grad[idxs] = grad
            return whole_t_grad

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)
