'''
Author: yitong 2969413251@qq.com
Date: 2023-02-16 18:03:08
'''
import numpy as np
import py_dl


seq_len = 96
dimension = 16
status_dimension = 12

signal_train, label_train, signal_test, label_test = py_dl.core.get_sequence_data(
    length=seq_len, dimension=dimension)

inputs = [py_dl.core.Variable(
    dim=(dimension, 1), init=False, trainable=False) for _ in range(seq_len)]

U = py_dl.core.Variable(dim=(status_dimension, dimension),
                        init=True, trainable=True)

W = py_dl.core.Variable(
    dim=(status_dimension, status_dimension), init=True, trainable=True)

b = py_dl.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

# store an arrayof internal state variables at each time
hiddens = []

last_step = None
for iv in inputs:
    h = py_dl.ops.Add(py_dl.ops.MatMul(U, iv), b)

    if last_step is not None:
        h = py_dl.ops.Add(py_dl.ops.MatMul(W, last_step), h)

    last_step = h
    hiddens.append(last_step)

# welding point, it doesn't connect a parent for now
welding_point = py_dl.ops.Welding()

fc1 = py_dl.layer.fc(welding_point, status_dimension, 40, "ReLU")
fc2 = py_dl.layer.fc(fc1, 40, 10, "ReLU")
output = py_dl.layer.fc(fc2, 10, 2, "None")

predict = py_dl.ops.Logistic(output)

label = py_dl.core.Variable((2, 1), trainable=False)

loss = py_dl.ops.CrossEntropyWithSoftMax(output, label)

learning_rate = 0.005
optimizer = py_dl.optimizer.Adam(py_dl.default_graph, loss, learning_rate)

batch_size = 16
curr_batch_size = 0

for epoch in range(30):
    for i, s in enumerate(signal_train):
        start = np.random.randint(len(s) // 3)
        end = np.random.randint(len(s) // 3 + 30, len(s))

        s = s[start: end]

        for j in range(len(s)):
            inputs[j].set_value(np.mat(s[j]).T)

        welding_point.weld(hiddens[j])

        label.set_value(np.mat(label_train[i, :]).T)

        optimizer.one_step()

        curr_batch_size += 1
        if curr_batch_size == batch_size:
            print(
                f"epoch: {epoch + 1}, iteration: {i + 1}, loss: {loss.value[0, 0]:.3f}")

            optimizer.update()
            curr_batch_size = 0

    pred = []

    for i, s in enumerate(signal_test):
        start = np.random.randint(len(s) // 3)
        end = np.random.randint(len(s) // 3 + 30, len(s))
        s = s[start:end]

        for j in range((len(s))):
            inputs[j].set_value(np.mat(s[j]).T)

        welding_point.weld(hiddens[j])

        predict.forward()
        pred.append(predict.value.A1)

    pred = np.array(pred).argmax(axis=1)
    true = label_test.argmax(axis=1)

    accuracy = (true == pred).sum() / len(signal_test)
    print(f"epoch: {epoch + 1}, accuracy: {accuracy:.5f}")
