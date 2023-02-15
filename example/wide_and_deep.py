'''
Author: yitong 2969413251@qq.com
Date: 2023-02-15 10:10:53
'''
import numpy as np
from sklearn.datasets import make_classification
import py_dl

dimension = 60

# 600 samples with dimension features. Binary sample is default. n_informative=20 means 20 features are useful among dimension features
X, y = make_classification(600, dimension, n_informative=20)
y = y * 2 - 1

# the dimension of embedding vectors
k = 20

# first order terms
x1 = py_dl.core.Variable(dim=(dimension, 1), init=False, trainable=False)

# label
label = py_dl.core.Variable(dim=(1, 1), init=False, trainable=False)

# weights of first order terms
w = py_dl.core.Variable(dim=(1, dimension), init=True, trainable=True)

# embedding matrix
E = py_dl.core.Variable(dim=(k, dimension), init=True, trainable=True)

# bias
b = py_dl.core.Variable(dim=(1, 1), init=True, trainable=True)

# the Wide part, a simple logistic regression
wide = py_dl.ops.MatMul(w, x1)

# the Deep part
# multiply the embedding matrices with the feature vector to get the embedding vector
embedding = py_dl.ops.MatMul(E, x1)

# the first hidden layer
hidden_1 = py_dl.layer.fc(embedding, k, 8, "ReLU")

# the second hidden layer
hidden_2 = py_dl.layer.fc(hidden_1, 8, 4, "ReLU")

# the output layer
deep = py_dl.layer.fc(hidden_2, 4, 1, None)

# the output
output = py_dl.ops.Add(wide, deep, b)

predict = py_dl.ops.Logistic(output)

# the loss function
loss = py_dl.ops.loss.LogLoss(py_dl.ops.Multiply(label, output))

learning_rate = 0.005

optimizer = py_dl.optimizer.Adam(py_dl.default_graph, loss, learning_rate)

batch_size = 16
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
        predict.forward()

        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5).astype(int) * 2 - 1
    accuracy = (y == pred).sum() / len(X)

    print(f"epoch: {epoch + 1}, accuracy: {accuracy:.3f}")
