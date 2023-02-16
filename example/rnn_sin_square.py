'''
Author: yitong 2969413251@qq.com
Date: 2023-02-16 11:36:32
'''
import numpy as np
import py_dl

# construct rnn
seq_len = 96  # the sequence length
dimension = 16  # the input dimension
status_dimension = 12  # the status dimension

signal_train, label_train, signal_test, label_test = py_dl.core.get_sequence_data(
    length=seq_len, dimension=dimension)

inputs = [py_dl.core.Variable(
    dim=(dimension, 1), init=False, trainable=False) for _ in range(seq_len)]


# the input weights matrix
U = py_dl.core.Variable(dim=(status_dimension, dimension),
                        init=True, trainable=True)

# the status weights matrix
W = py_dl.core.Variable(
    dim=(status_dimension, status_dimension), init=True, trainable=True)

# bias
b = py_dl.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

last_step = None  # last step input, the initial value is None
for iv in inputs:
    h = py_dl.ops.Add(py_dl.ops.MatMul(U, iv), b)

    if last_step is not None:
        h = py_dl.ops.Add(py_dl.ops.MatMul(W, last_step), h)

    h = py_dl.ops.ReLU(h)
    last_step = h

fc1 = py_dl.layer.fc(last_step, status_dimension, 40,
                     "ReLU")  # the first fc layer

fc2 = py_dl.layer.fc(fc1, 40, 10, "ReLU")
output = py_dl.layer.fc(fc2, 10, 2, "None")


predict = py_dl.ops.Logistic(output)

label = py_dl.core.Variable((2, 1), trainable=False)

loss = py_dl.ops.CrossEntropyWithSoftMax(output, label)

# train the model
learning_rate = 0.005
optimizer = py_dl.optimizer.Adam(py_dl.default_graph, loss, learning_rate)

batch_size = 16
curr_batch_size = 0

for epoch in range(20):
    for i, s in enumerate(signal_train):

        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)

        label.set_value(np.mat(label_train[i, :]).T)
        optimizer.one_step()

        curr_batch_size += 1
        if curr_batch_size == batch_size:
            print(
                f"epoch: {epoch + 1}, iteration: {i + 1}, loss: {loss.value[0, 0]}")
            optimizer.update()
            curr_batch_size = 0

    pred = []

    for i, s in enumerate(signal_test):

        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)

        predict.forward()

        pred.append(predict.value.A1)

    pred = np.array(pred).argmax(axis=1)
    true = label_test.argmax(axis=1)

    accuracy = (true == pred).sum() / len(signal_test)

    print(f"epoch: {epoch + 1}, accuracy: {accuracy:.3}")
