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


class Convolve(Operator):
    """treat the second parent as a filter, we take a two-dimensional discrete convolution of the first parent node"""

    def __init__(self, *parents, **kargs) -> None:
        assert len(parents) == 2
        Operator.__init__(self, *parents, **kargs)

        self.padded = None

    def compute(self):

        data = self.parents[0].value  # picture
        kernel = self.parents[1].value  # kernel

        w, h = data.shape  # width, height of the picture
        kw, kh = kernel.shape  # width, height of the kernel

        # the half of the width, height of the kernel
        hkw, hkh = int(kw / 2), int(kh / 2)
        # padding, so the kernel is a 'same kernel'
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))

        self.padded = np.mat(np.zeros((pw, ph)))
        self.padded[hkw:hkw + w, hkh:hkh + h] = data

        self.value = np.mat(np.zeros((w, h)))

        for i in np.arange(hkw, hkw + w):
            for j in np.arange(hkh, hkh + h):
                self.value[i - hkw, j - hkh] = np.sum(np.multiply(
                    self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh], kernel))

    def get_jacobi(self, parent) -> np.matrix:

        assert len(self.parents) == 2

        data = self.parents[0].value  # picture
        kernel = self.parents[1].value  # kernel

        w, h = data.shape  # the width and height of picture
        kw, kh = kernel.shape  # the width and height of kernel
        hkw, hkh = int(kw / 2), int(kh / 2)

        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))

        jacobi = []
        if parent is self.parents[0]:
            # some addition derivative and the multiplication derivative
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    mask = np.mat(np.zeros((pw, ph)))
                    mask[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh] = kernel
                    jacobi.append(mask[hkw:hkw + w, hkh:hkh + h].A1)
        else:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    jacobi.append(
                        self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh].A1)
        return np.mat(jacobi)


class ScalarMultiply(Operator):
    """multiply a matrix with a scalar"""

    def compute(self) -> None:
        assert self.parents[0].shape() == (
            1, 1)  # the first parent is a scalar
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent) -> np.matrix:
        assert parent in self.parents

        if parent is self.parents[0]:
            # numerator layout
            return self.parents[1].value.flatten().T
        else:
            return np.mat(np.eye(self.parents[1].dimension())) * self.parents[0].value[0, 0]


class MaxPooling(Operator):
    """max pooling operator"""

    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)

        self.stride = kargs.get('stride')
        assert isinstance(self.stride, tuple) and len(self.stride) == 2

        self.size = kargs.get('size')
        assert isinstance(self.size, tuple) and len(self.size) == 2

        self.flag = None

    def compute(self) -> None:
        data = self.parents[0].value  # the input feature maps
        w, h = data.shape  # the width, height of input feature maps
        dim = w * h
        sw, sh = self.stride
        kw, kh = self.size  # the size of the pooling kernel
        # the half of the width and height of the pooling kernel
        hkw, hkh = int(kw / 2), int(kh / 2)

        result = []
        flag = []

        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # set the boundaries of pooling kernel
                top, bottom = max(0, i - hkw), min(w, i + hkw + 1)
                left, right = max(0, j - hkh), min(h, j + hkh + 1)
                window = data[top:bottom, left:right]
                row.append(np.max(window))

                # record the position of the maximum in the original feature map
                pos = np.argmax(window)

                w_width = right - left
                offset_w, offset_h = top + pos // w_width, left + pos % w_width
                offset = offset_w * w + offset_h
                tmp = np.zeros(dim)
                tmp[offset] = 1
                flag.append(tmp)

            result.append(row)

        self.flag = np.mat(flag)
        self.value = np.mat(result)
    def get_jacobi(self, parent):
        assert parent is self.parents[0] and self.jacobi is not None
        return self.flag
    