'''
Author: yitong 2969413251@qq.com
Date: 2023-02-17 15:51:17
'''
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


def conv(feature_maps, input_shape, kernels, kernel_shape, activation):
    """
    :param feature_maps: list of feature maps, they are matrices with the same shape
    :param input_shape: tuple, width and height of feature map
    :param kernels: interger, the number of convolution kernel
    :param kernel_shape: tuple, the shape of kernel (width and height)
    :param activation: the activation function
    :return: array, include many featrue maps
    """
    # ones matrix with the same shape as the input matrix
    ones = Variable(input_shape, init=False, trainable=False)
    ones.set_value(np.mat(np.ones(input_shape)))

    outputs = []
    for _ in range(kernels):
        channels = []

        for fm in feature_maps:
            kernel = Variable(kernel_shape, init=True, trainable=True)
            conv = Convolve(fm, kernel)
            channels.append(conv)

        channels = Add(*channels)
        bias = ScalarMultiply(
            (Variable((1, 1), init=True, trainable=True)), ones)
        affine = Add(channels, bias)

        if activation == "ReLU":
            outputs.append(ReLU(affine))
        elif activation == "Logistic":
            outputs.append(Logistic(affine))
        else:
            outputs.append(affine)

    assert len(outputs) == kernels
    return outputs


def pooling(feature_maps, kernel_shape, stride):
    """
    :param feature_maps: array, include serveral intput feature maps, matrices with the same shape
    :param kernel_shape: tuple, the width and the height of the pooling kernel 
    :param stride: tuple, include horizontal and vertical stride 
    :return: array, include serveral output feature maps, matrices with the same shape
    """

    outputs = []
    for fm in feature_maps:
        outputs.append(MaxPooling(fm, size=kernel_shape, stride=stride))

    return outputs
