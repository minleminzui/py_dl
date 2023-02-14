'''
Author: yitong 2969413251@qq.com
Date: 2023-02-14 18:36:30
'''
import py_dl
import numpy as np
from sklearn.datasets import make_circles

X, y = make_circles(600, noise=0.1, factor=0.2)

y = y * 2 - 1

# feature dimensions
dimension = 20

# construct noise features
X = np.concatenate([X, np.random.normal(0.0, 0.01, (600, dimension - 2))], axis=1)

# hidden dimensions of features
k = 2

x1 = py_dl.core.Variable(dim=(dimension, 1), init=False, trainable=False)

label = py_dl.core.Variable(dim=(1, 1), init=False, trainable=False)

w = py_dl.core.Variable(dim=(1, dimension), init=True, trainable=True)

H = py_dl.core.Variable(dim=(k, dimension), init=True, trainable=True)

HTH = py_dl.ops.MatMul(py_dl.ops.Reshape(H, shape=(dimension, k)), H)

b = py_dl.core.Variable(dim=(1, 1), init=True, trainable=True)

output = py_dl.ops.Add(
    py_dl.ops.MatMul(w, x1),

    py_dl.ops.MatMul(py_dl.ops.Reshape(x1, shape=(1, dimension)),
                     py_dl.ops.MatMul(HTH, x1)),
                    b)

predict = py_dl.ops.Logistic(output)

loss = py_dl.ops.loss.LogLoss(py_dl.ops.Multiply(label, output))

learning_rate = 0.001

optimizer = py_dl.optimizer.Adam(py_dl.default_graph, loss, learning_rate)

batch_size = 16
curr_batch_size = 0

for epoch in range(20):

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
        predict.forward()

        pred.append(predict.value[0, 0])
    pred = (np.array(pred) > 0.5).astype(int) * 2 - 1

    accuracy = (y == pred).sum() / len(y)

    print(f"epoch: {epoch + 1}, accuracy: {accuracy:.3f}")


