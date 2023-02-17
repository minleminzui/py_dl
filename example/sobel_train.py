'''
Author: yitong 2969413251@qq.com
Date: 2023-02-17 14:50:25
'''
import py_dl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

pic = matplotlib.image.imread('data/lena.jpg') / 255

w, h = pic.shape

sobel = py_dl.core.Variable(dim=(3, 3), init=False, trainable=False)
sobel.set_value(np.mat([[1, 0, -1,], [2, 0, -2], [-1, 0, -1]]))

img = py_dl.core.Variable(dim=(w, h), init=False, trainable=False)
img.set_value(np.mat(pic))

sobel_output = py_dl.ops.Convolve(img, sobel)

sobel_output.forward()
plt.imshow(sobel_output.value, cmap="gray")

filter_train = py_dl.core.Variable(dim=(3, 3), init=True, trainable=True)
filter_output = py_dl.ops.Convolve(img, filter_train)

minus = py_dl.core.Variable(dim=(w, h), init=False, trainable=False)
minus.set_value(np.mat(-np.ones((w, h))))

n = py_dl.core.Variable((1, 1), init=False, trainable=False)
n.set_value(np.mat(1.0 / (w * h)))

error = py_dl.ops.Add(sobel_output, py_dl.ops.Multiply(filter_output, minus))
square_error = py_dl.ops.MatMul(py_dl.ops.Reshape(
    error, shape=(1, w * h)), py_dl.ops.Reshape(error, shape=(w * h, 1)))

mse = py_dl.ops.MatMul(square_error, n)

optimizer = py_dl.optimizer.Adam(py_dl.core.default_graph, mse, 0.01)

for i in range(1000):
    optimizer.one_step()
    optimizer.update()
    mse.forward()
    print(f"iteration: {i}, loss: {mse.value[0, 0]:.6f}")

filter_train.forward()
print(filter_train.value)

filter_output.forward()
plt.imshow(filter_output.value, cmap="gray")
plt.show()
