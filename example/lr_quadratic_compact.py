'''
Author: yitong 2969413251@qq.com
Date: 2023-02-14 12:44:52
'''
import sys
sys.path.append('..')
import numpy as np
import py_dl
from sklearn.datasets import make_circles


X, y = make_circles(200, noise=0.1, factor=0.2)
y = y * 2 - 1

x1 = py_dl.core.Variable(dim=(2, 1), init=False, trainable=False)

label = py_dl.core.Variable(dim=(1, 1), init=False, trainable=False)

w = py_dl.core.Variable(dim=(1, 2), init=True, trainable=True)

W = py_dl.core.Variable(dim=(2, 2), init=True, trainable=True)

b = py_dl.core.Variable(dim=(1, 1), init=True, trainable=True)


output = py_dl.ops.Add(
    py_dl.ops.MatMul(w, x1),  # one terms

    # quadratic term
    py_dl.ops.MatMul(py_dl.ops.Reshape(x1, shape=(1, 2)),
                     py_dl.ops.MatMul(W, x1)),
    b)

predict = py_dl.ops.Logistic(output)

loss = py_dl.ops.loss.LogLoss(py_dl.ops.Multiply(label, output))

learning_rate = 0.001

optimizer = py_dl.optimizer.Adam(py_dl.default_graph, loss, learning_rate)

batch_size = 8
curr_batch_size = 0

for epoch in range(50):

    for i in range(len(X)):
        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))

        optimizer.one_step()

        curr_batch_size += 1
        if curr_batch_size == batch_size:
            optimizer.update()
            curr_batch_size = 0

    pred = []

    for i in range(len(X)):
        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))

        predict.forward()
        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5).astype(int) * 2 - 1

    accuracy = (y == pred).sum() / len(y)

    print(f"epoch: {epoch + 1}, accuracy: {accuracy:.3f}")
