'''
Author: yitong 2969413251@qq.com
Date: 2023-02-09 14:22:04
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
                      np.concatenate((male_labels, female_labels))
                      ]).T

# randomly shuffle the samples
np.random.shuffle(train_set)

# constucting the computation graph: the input vector, a 3 * 1 matrix, without initialization, without training
x = py_dl.core.Variable(dim=(3, 1), init=False, trainable=False)

# gender label, 1 male, -1 female
label = py_dl.core.Variable(dim=(1, 1), init=False, trainable=False)

# weights vector, a 3 * 1 matrix, need to be initialized and participate in training
w = py_dl.core.Variable(dim=(1, 3), init=True, trainable=True)

# bias, a 1 * 1 matrix, need to be initialized and participate in training
b = py_dl.core.Variable(dim=(1, 1), init=True, trainable=True)

# the predictive output of ADALINE
output = py_dl.ops.Add(py_dl.ops.MatMul(w, x), b)
predict = py_dl.ops.Step(output)

# the loss function
loss = py_dl.ops.loss.PerceptionLoss(py_dl.ops.MatMul(label, output))

# learning rate
learning_rate = 0.0001

# training for 50 epoch
for epoch in range(50):

    # iterate over samples in the train data
    for i in range(len(train_set)):

        # take the first 3 columns of sample i (all columns except the last column) and construct a 3 * 1 matrix object
        # as a column vector
        features = np.mat(train_set[i, :-1]).T

        # take the last column of sample i as the sample label (1 male, -1 female) to construct a 1 * 1 matrix object
        l = np.mat(train_set[i, -1])

        # assign the feature to x node and assign the label to the label node
        x.set_value(features)
        label.set_value(l)

        # perform forawrd propagation on the loss node to calculate the loss value
        loss.forward()

        # perform backward propagation on w node and b node to calculate the jacobi matrices of the loss node for them
        w.backward(loss)
        b.backward(loss)

        """
        Update the parameter values with the loss values for the Jacobian of w and b (transpose of the gradient). The node that we want to optimize
        It should be a scalar node, and it should be the Jacobian of the variable node
        All shapes are 1 x n. The Jacobi transpose is the gradient of the result node to the variable node. Rearrange the gradient
        reshape into the shape of the variable matrix, and the corresponding position is the partial derivative of the result node with respect to the variable element.
        Multiply the gradient after changing the shape by the learning rate, subtract from the current variable value, and then assign the value to the variable node.
        The gradient descent update is complete.
        """
        # T means the gradient, do we really need it?
        w.set_value(w.value - learning_rate * w.jacobi.reshape(w.shape()))
        
        b.set_value(b.value - learning_rate * b.jacobi.reshape(b.shape()))

        # clear all the matries in nodes of computation graph
        py_dl.default_graph.clear_jacobi()

    # evaluate the accuracy of the model after every epoch
    pred = []

    # iterate over the training data, compute all the sample predications using current model
    for i in range(len(train_set)):

        features = np.mat(train_set[i, :-1]).T
        x.set_value(features)

        # perform forward propagation on the predict node of the model
        predict.forward()
        pred.append(predict.value[0, 0])

    # the output of step function is 0/1, we need convert them to 1/-1
    # cast pred as a np.array, so we can use the broadcasting
    pred = np.array(pred) * 2 - 1

    #
    accuracy = (train_set[:, -1] == pred).sum() / len(train_set)

    # print current epoch number and the accuracy of current model
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
