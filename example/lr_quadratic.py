'''
Author: yitong 2969413251@qq.com
Date: 2023-02-13 22:25:33
'''
import sys
sys.path.append("..")
from sklearn.datasets import make_circles
import py_dl
import numpy as np

X, y = make_circles(200, noise=0.1, factor=0.2)
y = y * 2 - 1

use_quadratic = True

x1 = py_dl.core.Variable(dim=(2, 1), init=False, trainiable=False)

label = py_dl.core.Variable(dim=(1, 1), init=False, trainiable=False)

b = py_dl.core.Variable(dim=(1, 1), init=True, trainiable=True)

# according to use_quadratic, differential deal
if use_quadratic:

    # multiply one terms times its own transpose to get the two terms to get a 2 * 2 matix, and then transpose it into a four dimension vector
    x2 = py_dl.ops.Reshape(py_dl.ops.MatMul(
        x1, py_dl.ops.Reshape(x1, shape=(1, 2))), shape=(4, 1))

    # concat one terms and two terms into a six dimension vector
    x = py_dl.ops.Concat(x1, x2)

    # weights vector is six dimension
    w = py_dl.core.Variable(dim=(1, 6), init=True, trainable=True)

else:

    x = x1

    w = py_dl.core.Variable(dim=(1, 2), init=True, trainable=True)

output = py_dl.ops.Add(py_dl.ops.MatMul(w, x), b)

predict = py_dl.ops.Logistic(output)

loss = py_dl.ops.loss.LogLoss(py_dl.ops.MatMul(label, output))

learning_rate = 0.001

optimizer = py_dl.optimizer.Adam(py_dl.default_graph, loss, learning_rate)

batch_size = 8
curr_batch_size = 0

for epoch in range(200):

    for i in range(len(X)):

        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))

        optimizer.one_step()

        curr_batch_size += 1

        if curr_batch_size >= batch_size:
            optimizer.update()
            curr_batch_size = 0

    pred = []
    for i in range(len(X)):
        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))

        predict.forward()
        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5).astype(int) * 2 - 1

    accuracy = (y == pred).astype(int).sum() / len(X)

    print(f"epoch: {epoch + 1}, accuracy: {accuracy:3f}")
