'''
Author: yitong 2969413251@qq.com
Date: 2023-02-07 16:22:38
'''
from __future__ import annotations
import numpy as np
import abc
from .graph import default_graph
from typing import List


class Node(abc.ABC):

    """computation graph node class base class"""

    def __init__(self, *parents) -> None:
        self.parents = parents  # the list of parent nodes
        self.children = []  # the list of children nodes
        self.value = None  # the value of this node
        self.jacobi = None  # the jacobi matrix for this node of the result node
        self.graph = default_graph  # default_graph is the default global computation graph

        # add this node to the children list of its parent node
        for parent in self.parents:
            parent.children.append(self)

        # add this node into the computation graph
        self.graph.add_node(self)

    def get_parents(self) -> List[Node]:
        """get the parent nodes of this node"""
        return self.parents

    def get_children(self) -> List[Node]:
        """get the children nodes of this node"""
        return self.children

    def gen_node_name(self, **kargs) -> None:
        """generate the name of this node
        if not sepcified by the users, generate a node similar to `MatMul:3` 
        based on the node type. If name_scope is specified, 
        a node name like `Hidden/MatMul:3` is generated
        """
        self.name = kargs.get('name', '{}:{}'.format(
            self.__class__.__name__, self.graph.node_count()))
        if self.graph.name_scope:
            self.name = '{}/{}'.format(self.graph.name_scope, self.name)

    def forward(self) -> None:
        """forward propagation to compute the value of this node
        if parent nodes are not computed, then invoking recursively the forward method of parent nodes
        """
        for node in self.parents:
            if node.value is None:
                node.forward()
        self.compute()

    # waring: the subclass of node must override the abstractmethods
    @abc.abstractmethod
    def compute(self):
        """abstract method, compute the value of this node based on its parent nodes"""

    @abc.abstractmethod
    def get_jacobi(self, parent):
        """abstract method, get the jacobi matrix of this node for some parent node
        this method will return ndarray, matrix and so on"""

    def backward(self, result) -> np.matrix:
        """backward propagation, compute the jacobi matrix of result node for this node"""
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                # assumer the dimesion of this node is column vector, so this jacobi matrix is the numerator layout
                self.jacobi = np.mat(
                    np.zeros((result.dimension(), self.dimension())))
                # derivation of addition and chain rule
                for child in self.get_children():
                    if child.value is not None:
                        self.jacobi += child.backward(result) * \
                            child.get_jacobi(self)
        return self.jacobi

    def clear_jacobi(self) -> None:
        """clear the jacobi matrix of the result node for this node"""
        self.jacobi = None

    def dimension(self) -> int:
        """return the dimension of this node which is flattened"""
        return self.value.shape[0] * self.value.shape[1]

    def shape(self) -> tuple:
        """return the shape of this node as a matrix: (row, column)"""
        return self.value.shape

    def reset_value(self, recursive=True) -> None:
        """reset the value of this node, and recursively reset the value of the downstream node of this node"""
        self.value = None

        if recursive:
            for child in self.children:
                child.reset_value()

class Variable(Node):
    """Variable Node"""

    def __init__(self, dim, init=False, trainable=True, **kargs) -> None:
        """
        Variable node do not have parent nodes
        the contructor accepts the dimension, 
        the identifier of whether to initialize 
        and the identifier of whether to participate to training
        """

        Node.__init__(self, **kargs)

        self.dim = dim

        # if this node need to be initialized, then initialize the value of this Variable randomly with a normal distribution
        # e.g. ms.core.Variable(dim=(3, 1), init=False, trainable=False)
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))

        # whether to participate to train
        self.trainable = trainable

    def set_value(self, value) -> None:
        """set the Variable value"""
        assert isinstance(value, np.matrix) and value.shape == self.dim

        # the value of this node changes, reset all the value of the downstream of this node
        self.reset_value()
        self.value = value

    def get_jacobi(self, parent):
        pass
    
    def compute(self):
        pass
