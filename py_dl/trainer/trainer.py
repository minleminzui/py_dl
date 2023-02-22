'''
Author: yitong 2969413251@qq.com
Date: 2023-02-22 14:15:03
'''
import abc
import numpy as np

import time
from ..core import Variable, default_graph


class Trainer(abc.ABC):
    """trainer"""

    def __init__(self, input_x, input_y, loss_op, optimizer, epoches, batch_size=8, eval_on_train=False, metrics_ops=None, *args, **kargs):

        # the input nodes of computation graph. there can be more than one. Their types are list
        self.inputs = input_x

        # the labels of the computation graph
        self.input_y = input_y

        # loss function
        self.loss_op = loss_op

        # optimizer
        self.optimizer = optimizer

        # epoches for training
        self.epoches = epoches
        self.epoch = 0

        # batch size
        self.batch_size = batch_size

        # whether to evaluate in traininig
        self.eval_on_train = eval_on_train

        # the list of evaluation metrics
        self.metrics_ops = metrics_ops

    def train_and_eval(self, train_x, train_y, test_x=None, test_y=None):
        """start training(evaluatation) process"""
        # initialize the weights variables
        self._variable_weights_init()

        # pass in the data to start the main loop
        self.main_loop(train_x, train_y, test_x, test_y)

    def main_loop(self, train_x, train_y, test_x, test_y):
        """train(evaluatation) main loop"""

        # the first loop, iterate epoches rounds
        for self.epoch in range(self.epoches):
            # train the model
            self.train(train_x, train_y)

            # if we need to eval the model
            if self.eval_on_train is not None and test_y is not None:
                self.eval(test_x, test_y)

    def train(self, train_x, train_y):
        """train the model with the training data"""
        # iterate over the training data
        # what is values()? train_x is a dict
        for i in range(len(list(train_x.values())[0])):
            # use a sample to perform a forward propagation and back propagation
            self.one_step(self._get_input_values(train_x, i,), train_y[i])

            # update the parameters
            if (i + 1) % self.batch_size == 0:
                self._optimizer_update()

    def eval(self, test_x, test_y):
        """evaluate the model with testing set"""
        for metrics_op in self.metrics_ops:
            metrics_op.reset_op()

        # iterate over the test set data
        for i in range(len(list(test_x.values())[0])):

            # perform the forward propagation of the computation graph and compute the evaluation metric
            self.one_step(self._get_input_values(
                test_x, i), test_y[i], is_eval=True)

            for metrcs_op in self.metrics_ops:
                metrcs_op.forward()

        # print the evaluation metric
        metrics_str = f'Epoch [{self.epoch + 1}] evaluation metrics'
        for metrics_op in self.metrics_ops:
            metrics_str += metrics_op.value_str()

        print(metrics_str)

    def _get_input_values(self, x, index):
        """x is the testing set of dict type, we need the element of the index position"""
        input_values = dict()
        for input_node_name in x.keys():
            input_values[input_node_name] = x[input_node_name][index]
        return input_values

    def one_step(self, data_x, data_y, is_eval=False):
        """perform a forward propagation and a backward propagation"""

        for i in range(len(self.inputs)):
            # get the corresponding data in the input data dict based on the name of the input node
            input_value = data_x.get(self.inputs[i].name)
            self.inputs[i].set_value(np.mat(input_value).T)

        # assign the label to the label nodes
        self.input_y.set_value(np.mat(data_y).T)

        # the optimizer is executed only during the training phase
        if not is_eval:
            self.optimizer.one_step()

    @abc.abstractmethod
    def _variable_weights_init(self):
        """weights variables initialization, the actual initialization is done by the subclass"""
        raise NotImplementedError()

    @abc.abstractmethod
    def _optimizer_update(self):
        """call optimizer to update parameters"""
        raise NotImplementedError()
