'''
Author: yitong 2969413251@qq.com
Date: 2023-02-12 14:31:27
'''

import numpy as np
import py_dl

male_heights = np.random.normal(171, 6, 500)
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [-1] * 500

train_set = np.array([np.concatenate((male_heights, female_heights)),
                      np.concatenate((male_weights, female_weights)),
                      np.concatenate((male_bfrs, female_bfrs)),
                      np.concatenate((male_labels, female_labels))]).T

np.random.shuffle(train_set)

x = py_dl.core.Variable(dim=(3, 1), init=False, trainable=False)

w = py_dl.core.Variable(dim=(1, 3), init=True, trainable=True)

label = py_dl.core.Variable(dim=(1, 1), init=False, trainable=False)

b = py_dl.core.Variable(dim=(1, 1), init=True, trainable=True)

output = py_dl.ops.Add(py_dl.ops.MatMul(w, x), b)
predict = py_dl.ops.Logistic(output)

loss = py_dl.ops.loss.LogLoss(py_dl.ops.MatMul(label, output))

learning_rate = 0.0001

optimizer = py_dl.optimizer.Adam(py_dl.default_graph, loss, learning_rate)

batch_size = 16
curr_batch_size = 0

for epoch in range(50):

    for i in range(len(train_set)): 

        features = np.mat(train_set[i, :-1]).T

        l = np.mat(train_set[i, -1])

        x.set_value(features)

        label.set_value(l)

        optimizer.one_step()

        curr_batch_size += 1

        if curr_batch_size >= batch_size:
            optimizer.update()
            curr_batch_size = 0
    
    pred = []

    for i in range(len(train_set)):

        features = np.mat(train_set[i, :-1]).T

        x.set_value(features)

        predict.forward()
        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5) * 2 - 1

    accuracy = (train_set[:, -1] == pred).sum() / len(train_set)
    print(f'epoch: {epoch + 1}, accuracy: {accuracy:.3f}')