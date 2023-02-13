'''
Author: yitong 2969413251@qq.com
Date: 2023-02-12 17:15:19
'''
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import py_dl

# read data as dataframe in pandas with dropping the Id column
data = pd.read_csv("../data/Iris.csv").drop("Id", axis=1)

# randomly shuffle samples order
data = data.sample(len(data), replace=False)

# transform Species label in string into intergers 0, 1, 2
le = LabelEncoder()
number_label = le.fit_transform(data["Species"])

# transform interger labels into One-Hot encoding
oh = OneHotEncoder(sparse_output=False)
one_hot_label = oh.fit_transform(number_label.reshape(-1, 1))

features = data[['SepalLengthCm',
                 'SepalWidthCm',
                 'PetalLengthCm',
                 'PetalWidthCm']]

x = py_dl.core.Variable(dim=(4, 1), init=False, trainable=False)

one_hot = py_dl.core.Variable(dim=(3, 1), init=False, trainable=False)

W = py_dl.core.Variable(dim=(3, 4), init=True, trainable=True)

b = py_dl.core.Variable(dim=(3, 1), init=True, trainable=True)

linear = py_dl.ops.Add(py_dl.ops.MatMul(W, x), b)

predict = py_dl.ops.SoftMax(linear)

loss = py_dl.ops.loss.CrossEntropyWithSoftMax(linear, one_hot)

learning_rate = 0.02

optimizer = py_dl.optimizer.Adam(py_dl.default_graph, loss, learning_rate)

batch_size = 16
curr_batch_size = 0

for epoch in range(200):

    for i in range(len(features)):

        feature = np.mat(features.iloc[i].values).T

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

        feature = np.mat(features.iloc[i].values).T
        x.set_value(feature)

        predict.forward()
        pred.append(predict.value.A.ravel())

    pred = np.array(pred).argmax(axis=1)
    # print(pred)
    accuracy = (number_label == pred).sum() / len(data)
       
    # 打印当前epoch数和模型在训练集上的正确率
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))