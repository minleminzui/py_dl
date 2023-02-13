'''
Author: yitong 2969413251@qq.com
Date: 2023-02-09 13:53:06
'''
import numpy as np

from ..core import Node
from .ops import SoftMax


class LossFunction(Node):
    """define Loss Function abstract class"""
    pass


class PerceptionLoss(LossFunction):
    """perceptionloss
    if the input is positive, it is 0
    if the input is negetive, it is the negative of the input"""

    def compute(self) -> None:
        self.value = np.mat(np.where(
            self.parents[0].value >= 0.0, 0.0, -self.parents[0].value))

    def get_jacobi(self, parent) -> np.ndarray:
        """
        jacobi matrix is a diagonal matrix
        """
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(diag.ravel())


class LogLoss(LossFunction):
    """Logistic loss function"""

    def compute(self) -> None:

        assert len(self.parents) == 1

        x = self.parents[0].value
        # prevent overflow
        self.value = np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))

    def get_jacobi(self, parent) -> None:

        x = parent.value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))

        return np.diag(diag.ravel())


class CrossEntropyWithSoftMax(LossFunction):
    """
    after applying SoftMax to the first patent node,
    cross entropy is calculated using the second parent node as One-Hot encoding 
    """

    def compute(self) -> None:
        prob = SoftMax.softmax(self.parents[0].value)
        # plus 1e-10 prevent grdient explosion
        self.value = np.mat(-np.sum(np.multiply(
            self.parents[1].value, np.log(prob + 1e-10))))

    def get_jacobi(self, parent):

        prob = SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T
