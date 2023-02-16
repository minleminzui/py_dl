'''
Author: yitong 2969413251@qq.com
Date: 2023-02-15 16:53:59
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import py_dl

data = pd.read_csv(
    "data/titanic.csv").drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

le = LabelEncoder()
oh = OneHotEncoder(sparse_output=False)

Pclass = oh.fit_transform(le.fit_transform(
    data["Pclass"].fillna(0)).reshape(-1, 1))

Sex = oh.fit_transform(le.fit_transform(data["Sex"].fillna("")).reshape(-1, 1))

Embarked = oh.fit_transform(le.fit_transform(
    data["Embarked"].fillna("")).reshape(-1, 1))


# double '['?
features = np.concatenate(
    [Pclass, Sex, data[["Age"]].fillna(0), data[["SibSp"]].fillna(0), data[["Parch"]].fillna(0), data[["Fare"]].fillna(0), Embarked], axis=1)

labels = data["Survived"].values * 2 - 1

dimension = features.shape[1]

k = 12

x1 = py_dl.core.Variable(dim=(dimension, 1), init=False, trainable=False)

label = py_dl.core.Variable(dim=(1, 1), init=False, trainable=False)

w = py_dl.core.Variable(dim=(1, dimension), init=True, trainable=True)

H = py_dl.core.Variable(dim=(k, dimension), init=True, trainable=True)
HTH = py_dl.ops.MatMul(py_dl.ops.Reshape(H, shape=(dimension, k)), H)

b = py_dl.core.Variable(dim=(1, 1), init=True, trainable=True)

output = py_dl.ops.Add(py_dl.ops.MatMul(w, x1),
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

    for i in range(len(features)):
        x1.set_value(np.mat(features[i]).T)
        label.set_value(np.mat(labels[i]))

        optimizer.one_step()
        curr_batch_size += 1
        if curr_batch_size == batch_size:
            optimizer.update()
            curr_batch_size = 0

    pred = []

    for i in range(len(features)):
        x1.set_value(np.mat(features[i]).T)
        predict.forward()

        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5).astype(int) * 2 - 1
    accuracy = (labels == pred).sum() / len(features)

    print(f"epoch: {epoch + 1}, accuracy: {accuracy:.3f}")
