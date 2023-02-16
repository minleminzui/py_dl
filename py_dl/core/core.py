'''
Author: yitong 2969413251@qq.com
Date: 2023-02-09 19:47:56
'''
from .node import Variable
from .graph import default_graph
import numpy as np
from scipy import signal  


def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None


# construct sine wave and square wave
def get_sequence_data(dimension=10, length=10, number_of_example=1000, train_set_ratio=0.7, seed=42):
    """generate two kinds of sequence data"""
    xx = []

    xx.append(np.sin(np.arange(0, 10, 10 / length)).reshape(-1, 1))

    xx.append(np.array(signal.square(
        np.arange(0, 10, 10 / length))).reshape(-1, 1))

    data = []

    for i in range(2):
        x = xx[i]
        for _ in range(number_of_example // 2):
            # what noise?
            sequence = x + \
                np.random.normal(0, 0.6, (len(x), dimension))  # Add noise
            # one-hot label
            label = np.array([int(i == k) for k in range(2)])
            data.append(np.c_[sequence.reshape(1, -1), label.reshape(1, -1)])

    # mix up the samples from each category
    data = np.concatenate(data, axis=0)

    # randomly shuffle the sample order
    np.random.shuffle(data)

    # compute the number of the samples
    train_set_size = int(number_of_example * train_set_ratio)

    return (data[:train_set_size, :-2].reshape(-1, length, dimension),
            data[:train_set_size, -2:],
            data[train_set_size:, :-2].reshape(-1, length, dimension),
            data[train_set_size:, -2:])
