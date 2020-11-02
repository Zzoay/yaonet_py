
import numpy as np

from typing import List, Callable, Union, Tuple


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
        self.shape = self._data.shape
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

    def reshape(self, *shape):
        return BasicTensor(self.data.reshape(*shape))

    def ravel(self):
        return BasicTensor(self.data.ravel())


# a tensor includes data and dependency info.
class Tensor(BasicTensor):
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = []) -> None:
        super().__init__(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on
        self.grad: np.ndarray = None
      
    @property
    def data(self) -> np.ndarray:
        return super().data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        super(Tensor, Tensor).data.__set__(self, new_data)
        # set a tensor means that its grad should become None.
        self.grad = None

    def reshape(self, *shape):
        return _reshape(self, *shape)

    # TODO define backward pass
    def ravel(self):
        raise NotImplementedError

    def squeeze(self, axis):
        shape = list(self.shape).pop(axis)
        return _reshape(self, shape)

    def unsqueeze(self, axis):
        shape = list(self.shape).insert(axis, 1)
        return _reshape(self, shape)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def sum(self) -> 'Tensor':
        return _sum(self)

    # t + other
    def __add__(self, other) -> 'Tensor':
        return _add(self, ensure_tensor(other))

    # other + t
    def __radd__(self, other) :
        return _add(ensure_tensor(other), self)
    
    # t = t + other
    def __iadd__(self, other):
        self.data = self.data + ensure_tensor(other).data
        return self

    def __mul__(self, other):
        return _mul(self, ensure_tensor(other))
    
    def __rmul__(self, other) :
        return _mul(ensure_tensor(other), self)
    
    def __imul__(self, other):
        self.data = self.data * ensure_tensor(other).data
        return self

    def __neg__(self):
        return _neg(self)

    def __sub__(self, other):
        return _sub(self, ensure_tensor(other))
    
    def __rsub__(self, other):
        return _sub(ensure_tensor(other), self)

    def __isub__(self, other):
        self.data = self.data - ensure_tensor(other).data
        return self

    def __matmul__(self, other):
        return _matmul(self, ensure_tensor(other))

    def __truediv__(self, other):
        return _div(self, ensure_tensor(other))

    def __rtruediv__(self, other):
        return _div(ensure_tensor(other), self)
    
    def __itruediv__(self, other):        
        self.data = self.data / ensure_tensor(other).data
        return self

    def __getitem__(self, index):
        return _slice(self, index)
        
    def backward(self, grad: np.ndarray = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if self.grad is None:
            self.zero_grad()

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


def _sum(t: Tensor) -> Tensor:
    data: Tensor = t.data.sum()
    requires_grad: bool = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # the gradient of function 'sum' is 1 to all elements
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data: Tensor = t1.data + t2.data
    requires_grad: bool = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # sum out added dims
            ndims_added = grad.ndim - t1.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # sum out added dims
            ndims_added = grad.ndim - t2.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    data: Tensor = t1.data * t2.data
    requires_grad: bool = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # chain rule
            grad = grad * t2.data

            # sum out added dims
            ndims_added = grad.ndim - t1.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # chain rule
            grad = grad * t1.data

            # sum out added dims
            ndims_added = grad.ndim - t2.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(
            data,
            requires_grad,
            depends_on)


def _div(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data / t2.data
    requires_grad: bool = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # chain rule
            grad = grad * (1 / t2.data)

            # sum out added dims
            ndims_added = grad.ndim - t1.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # chain rule
            grad = grad * t1.data * (-1 / t2.data**2)

            # sum out added dims
            ndims_added = grad.ndim - t2.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(
            data,
            requires_grad,
            depends_on)


def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad

    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2


def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    t1shape = t1.shape
    t2shape = t2.shape

    if len(t1shape) > 3 or len(t2shape) > 3:
        raise RuntimeError("matmul operation only supports 1-3D tensors now, you can use tensordot.")

    data = np.matmul(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # where t3 = t1 @ t2 (@ means matmul), t1 is (a ,b), t2 is (b, c) -> t3 (a, c)
            # chain rule: grad1 = grad3 @ t2.T
            if len(t2shape) != 3:
                return np.matmul(grad, t2.data.T)  # 'grad @ t2.data.T' also work
            else:  # means a tensor, always [batch_size, height, width], just reverse the dims apart from 1st.
                return np.matmul(grad, t2.data.reshape(-1, *t2shape[:0:-1]))  # 

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # chain rule: grad2 = t1.T @ grad3
            if len(t2shape) != 3:  # means a vector or matrix, just dot 0 to the second last dims.
                n = list(range(0, len(t1shape[:-1])))
            else:  # means a tensor, always [batch_size, height, width], dot is not needed for the first dimension.
                n = list(range(1, len(t1shape[:-1])))

            return np.tensordot(t1.data, grad, axes=(n, n))

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


# TODO unittest
def _tensordot(t1: Tensor, t2: Tensor, dims: Union[int, Tuple[List[int]]]) -> Tensor:
    t1shape = list(t1.shape)
    t2shape = list(t2.shape)

    t1od = [i for i in range(len(t1shape)) if i not in dims[0]]
    t2od = [i for i in range(len(t2shape)) if i not in dims[1]]

    partition = len(t1od)

    data = np.tensordot(t1.data, t2.data, dims)
    dshape = list(data.shape)

    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return np.tensordot(t2.data, grad, axes=(t2od, list(range(partition, len(dshape))))).reshape(t1shape)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return np.tensordot(t1.data, grad, axes=(t1od, list(range(partition)))).reshape(t2shape)

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)


def _slice(t: Tensor, idxs: Union[int, List[int]]) -> Tensor:
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


def _reshape(t: Tensor, *shape):
    pre_shape = t.shape

    data = t.data.reshape(*shape)
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad.reshape(*pre_shape)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)
