'''
Author: yitong 2969413251@qq.com
Date: 2023-02-09 13:53:06
'''
import numpy as np

from ..core import Node


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
