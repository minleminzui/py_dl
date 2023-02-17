'''
Author: yitong 2969413251@qq.com
Date: 2023-02-17 11:54:29
'''
import py_dl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# read image, normalization
pic = matplotlib.image.imread('data/mondrian.jpg') / 255

# size of the image
w, h = pic.shape

# longitudinal sobel filter
sobel_v = py_dl.core.Variable(dim=(3, 3), init=False, trainable=False)
sobel_v.set_value(np.mat([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))

# tranverse sobel filter
sobel_h = py_dl.core.Variable(dim=(3, 3), init=False, trainable=False)
sobel_h.set_value(sobel_v.value.T)

# the input image
img = py_dl.core.Variable(dim=(w, h), init=False, trainable=False)

img.set_value(np.mat(pic))

# the output of sobel filter
sobel_v_output = py_dl.ops.Convolve(img, sobel_v)
sobel_h_output = py_dl.ops.Convolve(img, sobel_h)

# the sum of squares of two sobel filters
square_output = py_dl.ops.Add(
    py_dl.ops.Multiply(sobel_v_output, sobel_v_output),
    py_dl.ops.Multiply(sobel_h_output, sobel_h_output))

square_output.forward()

# output img
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(221)
ax.axis("off")
ax.imshow(img.value, cmap="gray")

ax = fig.add_subplot(222)
ax.axis("off")
ax.imshow(square_output.value, cmap="gray")

ax = fig.add_subplot(223)
ax.axis("off")
ax.imshow(sobel_v_output.value, cmap="gray")

ax = fig.add_subplot(224)
ax.axis("off")
ax.imshow(sobel_h_output.value, cmap="gray")

plt.savefig('img/sobel_output.png')
