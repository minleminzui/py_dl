'''
Author: yitong 2969413251@qq.com
Date: 2023-02-20 10:41:31
'''
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.io import loadmat
import py_dl

# get minist dataset, and take some samples and normalize them
mnist = loadmat('data/mnist-original.mat')
X, y = mnist['data'].T, mnist['label']
np.random.shuffle(X)
np.random.shuffle(y)
X, y = X[:1000] / 255, y.astype(int)[:1000]

# transform interger labels into One-Hot encoding
oh = OneHotEncoder(sparse_output=False)
one_hot_label = oh.fit_transform(y.reshape(-1, 1))

# intput image size
img_shape = (28, 28)

# input image
x = py_dl.core.Variable(img_shape, init=False, trainable=False)

# One-Hot label
one_hot = py_dl.core.Variable(dim=(10, 1), init=False, trainable=False)

# the first cnn layer
conv1 = py_dl.layer.conv([x], img_shape, 3, (5, 5), "ReLU")

# the first maxpooling layer
pooling1 = py_dl.layer.pooling(conv1, (3, 3), (2, 2))

# the second cnn layer
conv2 = py_dl.layer.conv(pooling1, (14, 14), 3, (3, 3), "ReLU")

# the second maxpooling layer
pooling2 = py_dl.layer.pooling(conv2, (3, 3), (2, 2))

# the fullconnected layer
fc1 = py_dl.layer.fc(py_dl.ops.Concat(*pooling2), 147, 120, "ReLU")

# the output layer
output = py_dl.layer.fc(fc1, 120, 10, "None")

predict = py_dl.ops.SoftMax(output)
# crossentropyloss
loss = py_dl.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

# learning rate
learning_rate = 0.005

# optimizer
optimizer = py_dl.optimizer.Adam(py_dl.default_graph, loss, learning_rate)

# batch size
batch_size = 32
curr_batch_size = 0

for epoch in range(50):
    for i in range(len(X)):
        feature = np.mat(X[i]).reshape(img_shape)
        label = np.mat(one_hot_label[i]).T

        x.set_value(feature)
        one_hot.set_value(label)

        optimizer.one_step()

        curr_batch_size += 1
        if curr_batch_size == batch_size:
            print(
                f"epoch: {epoch + 1}, iteration: {i + 1}, loss: {loss.value[0, 0]}")
            optimizer.update()
            curr_batch_size = 0

    pred = []
    for i in range(len(X)):
        feature = np.mat(X[i]).reshape(img_shape)
        x.set_value(feature)

        predict.forward()
        pred.append(predict.value.A1)

    pred = np.array(pred).argmax(axis=1)

    accuracy = (y == pred).sum() / len(X)
    print(f"epoch: {epoch + 1}, accuracy: {accuracy}")
