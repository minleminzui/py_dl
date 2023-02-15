'''
Author: yitong 2969413251@qq.com
Date: 2023-02-15 13:55:24
'''
import numpy as np
from sklearn.datasets import make_classification
import py_dl

dimension = 60

X, y = make_classification(600, dimension, n_informative=20)
y = y * 2 - 1

k = 20

x1 = py_dl.core.Variable(dim=(dimension, 1), init=False, trainable=False)

label = py_dl.core.Variable(dim=(1, 1), init=False, trainable=False)

w = py_dl.core.Variable(dim=(1, dimension), init=True, trainable=True)

E = py_dl.core.Variable(dim=(k, dimension), init=True, trainable=True)

b = py_dl.core.Variable(dim=(1, 1), init=True, trainable=True)

# embedding vector
embedding = py_dl.ops.MatMul(E, x1)

# the FM part
fm = py_dl.ops.Add(py_dl.ops.MatMul(w, x1),
                   py_dl.ops.MatMul(py_dl.ops.Reshape(embedding, shape=(1, k)), embedding))

# the Deep part, the first hidden layer
hidden_1 = py_dl.layer.fc(embedding, k, 8, "ReLU")

# the second hidden layer
hidden_2 = py_dl.layer.fc(hidden_1, 8, 4, "ReLU")

deep = py_dl.layer.fc(hidden_2, 4, 1, None)

output = py_dl.ops.Add(deep, fm, b)

predict = py_dl.ops.Logistic(output)

loss = py_dl.ops.loss.LogLoss(py_dl.ops.Multiply(label, output))

learning_rate = 0.005
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
