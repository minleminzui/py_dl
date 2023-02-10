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
