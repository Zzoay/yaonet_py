
import numpy as np

from yaonet.tensor import Dependency, Tensor, _add, _sub, _mul, _matmul, _div, ensure_tensor


def add(t1: Tensor, t2: Tensor) -> Tensor:
    return _add(ensure_tensor(t1), ensure_tensor(t2))


def mul(t1: Tensor, t2: Tensor) -> Tensor:
    return _mul(ensure_tensor(t1), ensure_tensor(t2))


def sub(t1: Tensor, t2: Tensor) -> Tensor:
    return _sub(ensure_tensor(t1), ensure_tensor(t2))


def div(t1: Tensor, t2: Tensor) -> Tensor: 
    return _div(ensure_tensor(t1), ensure_tensor(t2))


def matmul(t1: Tensor, t2: Tensor) -> Tensor:
    return _matmul(ensure_tensor(t1), ensure_tensor(t2))


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
