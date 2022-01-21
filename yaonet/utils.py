
from typing import Tuple, Union, Iterable, List

import numpy as np

from yaonet.tensor import Tensor, Dependency


# previous version
def _cat(ts: Tuple[Tensor], axis: int) -> Tensor:
    _data = list()
    requires_grad = False
    depends_on = []

    offset = 0
    offsets = []
    for t in ts:
        _data.append(t.data)
        if t.requires_grad:
            requires_grad = True

            tmp = t.shape[axis]
            h = offset+tmp
            offsets.append((offset, h))
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                _l, _h = next(offsets_iter)
                return grad.take(list(range(_l, _h)), axis=axis)

            offset = h
            depends_on.append(Dependency(t, grad_fn))

    offsets_iter = iter(offsets)
    data = np.concatenate(_data, axis=axis)
    del _data

    return Tensor(data, requires_grad, depends_on)


def cat(ts: Union[List[Tensor], Tuple[Tensor, ...]], axis: int) -> Tensor:
    _data = list()
    requires_grad = False
    depends_on = []

    offset = 0
    offsets = []
    for t in ts:
        _data.append(t.data)
        if t.requires_grad:
            requires_grad = True

            tmp = t.shape[axis]
            h = offset+tmp
            offsets.append((offset, h))

            def make_func(offset, h):
                def grad_fn(grad: np.ndarray) -> np.ndarray:
                    _l, _h = offset, h
                    return grad.take(list(range(_l, _h)), axis=axis)
                return grad_fn

            depends_on.append(Dependency(t, make_func(offset, h)))
            offset = h

    data = np.concatenate(_data, axis=axis)
    del _data

    return Tensor(data, requires_grad, depends_on)


def clip_grad_value(parameters: Union[Tensor, Iterable[Tensor]], 
                    clip_value: float) -> None:
    assert clip_value > 0, "clip_value must be > 0"
    for parameter in parameters:  # type: ignore
        parameter.grad = np.clip(parameter.grad, -clip_value, clip_value)   # type: ignore
    return