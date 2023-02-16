'''
Author: yitong 2969413251@qq.com
Date: 2023-02-08 09:25:22
'''

import numpy as np

from ..core import Node


def fill_diagonal(to_be_filled, filler) -> np.matrix:
    """fill filler on the diagonal of to_be_filled generally"""
    assert to_be_filled.shape[0] / \
        filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]

    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled


class Operator(Node):
    """define the Operator abstract class"""
    pass


class Add(Operator):
    """(multiple) matries addition"""

    def compute(self) -> None:
        self.value = np.mat(np.zeros(self.parents[0].shape()))

        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent) -> np.matrix:
        # the jacobi matrix of the sum of all the matrices for any matrix is the identity matrix
        return np.mat(np.eye(parent.dimension()))


class MatMul(Operator):
    """Matrix multiplication"""

    def compute(self) -> None:
        assert len(self.parents) == 2 and self.parents[0].shape()[
            1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[1].value

    def get_jacobi(self, parent) -> np.matrix:
        """consider the matrix multiplication as mapping and to compute the jacobi matrix of the mapping for the involved matrices"""
        # see p73 of the book about MatrixSlow
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(
                self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(
                parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


class Step(Operator):
    """step function, a sort of activation function"""

    def compute(self) -> None:
        self.value = np.mat(np.where(self.parents[0].value >= 0.0, 1.0, 0.0))

    def get_jacobi(self, parent) -> np.matrix:
        return np.mat(np.zeros(parent.dimensions()))


class Logistic(Operator):
    """apply a Logistic function to components of the vector"""

    def compute(self):
        assert len(self.parents) == 1

        x = self.parents[0].value
        self.value = np.mat(
            1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x))))

    def get_jacobi(self, parent) -> np.matrix:
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)


class SoftMax(Operator):
    """Softmax function"""

    # we will use this function in other place, so we set it as staticmethod
    @staticmethod
    def softmax(a):
        a[a > 1e2] = 1e2  # prevent excessive exponent
        ep = np.power(np.e, a)
        return ep / np.sum(ep)

    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        """
        we do not use the get_jacobi function of SoftMax node
        we use CrossEntropyWithSoftMax instead when training
        """
        raise NotImplementedError("Don't use SoftMax's get_jacobi")


class ReLU(Operator):
    """assign ReLU function to elements in the martrix"""

    nslope = 0.1  # the slope of the negative axis

    def compute(self) -> None:
        self.value = np.mat(np.where(
            self.parents[0].value > 0.0, self.parents[0].value, self.nslope * self.parents[0].value))

    def get_jacobi(self, parent) -> np.matrix:
        return np.diag(np.where(parent.value.A1 > 0.0, 1.0, self.nslope))


class Reshape(Operator):
    """change the shape of the parent matrix"""

    def __init__(self, *parent, **kargs) -> None:
        Operator.__init__(self, *parent, **kargs)

        self.to_shape = kargs.get('shape')
        assert isinstance(self.to_shape, tuple)

    def compute(self) -> None:
        self.value = self.parents[0].value.reshape(self.to_shape)

    def get_jacobi(self, parent) -> None:
        assert parent is self.parents[0]

        return np.mat(np.eye(self.dimension()))


class Concat(Operator):
    """concat multiple parents into a single vector"""

    def compute(self):
        assert len(self.parents) > 0

        self.value = np.concatenate([p.value.flatten()
                                    for p in self.parents], axis=1).T

    def get_jacobi(self, parent):
        assert parent in self.parents

        dimensions = [p.dimension() for p in self.parents]
        pos = self.parents.index(parent)

        dimension = parent.dimension()

        assert dimension == dimensions[pos]

        jacobi = np.mat(np.zeros((self.dimension(), dimension)))
        start_row = int(np.sum(dimensions[:pos]))
        jacobi[start_row:start_row + dimension,
               0:dimension] = np.eye(dimension)

        return jacobi


class Multiply(Operator):
    """two parents matrices with the same shape, multiply in elementwise"""

    def compute(self) -> None:
        assert self.parents[0].value.shape == self.parents[1].value.shape
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent) -> np.ndarray:
        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A1)
        else:
            return np.diag(self.parents[0].value.A1)


class Welding(Operator):

    def compute(self) -> None:
        assert len(self.parents) == 1 and self.parents[0] is not None

        self.value = self.parents[0].value

    def get_jacobi(self, parent) -> np.matrix:
        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))

    def weld(self, node):
        """weld this node to the input node"""
        if len(self.parents) == 1 and self.parents[0] is not None:
            self.parents[0].children.remove(self)

        self.parents.clear()

        # weld
        self.parents.append(node)
        node.children.append(self)
