from ..core import *
from ..ops import *


def fc(input, input_size, size, activation) -> Operator:
    '''
    :param input: input vector
    :param input_size: dimension of input vector
    :param size: size of neuron, i.e. the size of output (the dimension of the output vector)
    :param activation: type of activation function
    :return: the output vector
    '''

    weights = Variable((size, input_size), init=True, trainable=True)
    bias = Variable((size, 1), init=True, trainable=True)
    affine = Add(MatMul(weights, input), bias)

    if activation == "ReLU":
        return ReLU(affine)
    elif activation == "Logistic":
        return Logistic(affine)
    else:
        return affine
