from yaonet.tensor import Tensor


class Loss():
    
    def loss(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return self.loss(predicted, actual)


class MSE(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> Tensor:
        errors = (predicted - actual)
        return (errors*errors).sum()