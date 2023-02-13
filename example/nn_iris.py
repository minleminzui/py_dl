'''
Author: yitong 2969413251@qq.com
Date: 2023-02-13 15:45:49
'''
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import py_dl

# data = pd.read_csv("../data/Iris.csv").drop("Id", axis=1)
data = pd.read_csv("../data/Iris.csv").drop("Id", axis=1)
data = data.sample(len(data), replace=False)


le = LabelEncoder()
number_label = le.fit_transform(data["Species"])

oh = OneHotEncoder(sparse_output=False)
one_hot_label = oh.fit_transform(number_label.reshape(-1, 1))

features = data[['SepalLengthCm',
                 'SepalWidthCm',
                 'PetalLengthCm',
                 'PetalWidthCm']].values

x = py_dl.core.Variable(dim=(4, 1), init=False, trainable=False)

one_hot = py_dl.core.Variable(dim=(3, 1), init=False, trainable=False)

hidden_1 = py_dl.layer.fc(x, 4, 10, "ReLU")

hidden_2 = py_dl.layer.fc(hidden_1, 10, 10, "ReLU")

output = py_dl.layer.fc(hidden_2, 10, 3, None)

predict = py_dl.ops.SoftMax(output)

loss = py_dl.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

learning_rate = 0.02

optimizer = py_dl.optimizer.Adam(py_dl.default_graph, loss, learning_rate)

batch_size = 16
curr_batch_size = 0

for epoch in range(30):

    for i in range(len(features)):

        feature = np.mat(features[i, :]).T

        label = np.mat(one_hot_label[i, :]).T

        x.set_value(feature)
        one_hot.set_value(label)

        optimizer.one_step()

        curr_batch_size += 1
        if curr_batch_size == batch_size:
            optimizer.update()
            curr_batch_size = 0

    pred = []

    for i in range(len(features)):
        feature = np.mat(features[i, :]).T
        x.set_value(feature)

        predict.forward()
        # we should append the array, or else we will get the bigger sum
        pred.append(predict.value.A1)

    pred = np.array(pred).argmax(axis=1)

    accuracy = (number_label == pred).astype(int).sum() / len(data)

    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1,
                                                 accuracy))
