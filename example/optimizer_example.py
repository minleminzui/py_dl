'''
Author: yitong 2969413251@qq.com
Date: 2023-02-11 17:07:24
'''
import sys
sys.path.append('..')
import numpy as np
import py_dl

"""
Manufacturing training samples. Sampling 500 male heights for a normal distribution of mean 171 and standard deviation 6.
Sampling 500 female heights for a normal distribution of mean 158 and standard deviation 5. Sampling 500 male weights for
a normal distribution of mean 70 and standard deviation 10. Sampling 500 female weights for a normal distribution of mean
57 and standard deviation 8. Sampling 500 male boady fat rate for a normal distribution of mean 16 and standard deviation 2.
Sampling 500 female body fat rate for a normal distribution of mean 22 and standard deviation 2. Constructing 500 Ones, as 
the male label. Construct 500 minus ones, as the female label. Assembling the data into a 100 * 4 numpy array. The frist three column
respectively are heights, weights and body fat rate. The last column is the gender label.
"""

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


label = py_dl.core.Variable(dim=(1, 1), init=False, trainable=False)

w = py_dl.core.Variable(dim=(1, 3), init=True, trainable=True)

b = py_dl.core.Variable(dim=(1, 1), init=True, trainable=True)

output = py_dl.ops.Add(py_dl.ops.MatMul(w, x), b)

predict = py_dl.ops.Step(output)

loss = py_dl.ops.loss.PerceptionLoss(py_dl.ops.MatMul(label, output))

learning_rate = 0.01

# use all kinds of optimizers
# optimizer = py_dl.optimizer.GradientDescent(py_dl.default_graph, loss, learning_rate)
optimizer = py_dl.optimizer.Momentum(py_dl.default_graph, loss, learning_rate)
# optimizer = py_dl.optimizer.AdaGrad(py_dl.default_graph, loss, learning_rate)
# optimizer = py_dl.optimizer.RMSProp(py_dl.default_graph, loss, learning_rate)
# optimizer = py_dl.optimizer.Adam(py_dl.default_graph, loss, learning_rate)

mini_batch_size = 8
cur_batch_size = 0


for epoch in range(50):

    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T

        l = np.mat(train_set[i, -1])

        x.set_value(features)
        label.set_value(l)

        optimizer.one_step()
        cur_batch_size += 1
        if (cur_batch_size == mini_batch_size):
            optimizer.update()
            cur_batch_size = 0

    pred = []

    for i in range(len(train_set)):

        features = np.mat(train_set[i, :-1]).T

        x.set_value(features)
        predict.forward()
        pred.append(predict.value[0, 0])

    pred = np.array(pred) * 2 - 1

    accuracy = (train_set[:, -1] == pred).sum() / len(train_set)

    print(f"epoch: {epoch + 1}, accuracy: {accuracy:.3f}")
