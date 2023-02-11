'''
Author: yitong 2969413251@qq.com
Date: 2023-02-10 13:14:53
'''
import abc

import numpy as np

from ..core import Node, Variable, Graph, get_node_from_graph


class Optimizer(abc.ABC):
    """Optimizer class"""

    def __init__(self, graph, target, learning_rate=0.01) -> None:
        """
        Optimizer constructor accepts computation graph object, target node object and learning rate
        """
        assert isinstance(target, Node) and isinstance(graph, Graph)

        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate

        # accumulate all the gradients of a mini batch of samples for every training node
        self.acc_gradient = dict()
        self.acc_no = 0

    def one_step(self) -> None:
        """compute and accumulate the gradient of samples"""
        self.forward_backward()
        self.acc_no += 1

    def get_gradient(self, node) -> float:
        """
        return the average gradient of samples
        """
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no

    @abc.abstractmethod
    def _update(self) -> None:
        """
        abstract method to perform the specific algorithm of updating gradient implemented by subclasses
        """

    def update(self) -> None:

        # update gradients
        self._update()

        # clear accumulated gradients
        self.acc_gradient.clear()
        self.acc_no = 0

    def forward_backward(self) -> None:
        """forward propagation to update values of nodes and backward propagation jacobi matrices of  the result node for every nodes"""

        # clear all jacobi matrices of all nodes in the computation graph
        self.graph.clear_jacobi()

        # forward propagation to compute the result node
        self.target.forward()

        # backward propagation to compute the jacobi matrices
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)

                # the jacobi matrix of the result node for this node is a row vector, and its transposition is gradient(column vector)
                # reshape the gradient as the same shape with this node which is convenient for updating the value of this node
                gradient = node.jacobi.reshape(node.shape())
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    self.acc_gradient[node] += gradient


class GradientDescent(Optimizer):
    """Grident Descent Optimizer"""
    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target, learning_rate)

    def _update(self) -> None:
        """plain gradient descent"""
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                # gets the average gradient of current batch of this node
                gradient = self.get_gradient(node)
                # updates the value of variable nodes using plain gradient descent
                node.set_value(node.value - self.learning_rate * gradient)


class Momentum(Optimizer):
    """Momentum optimizer"""

    def __init__(self, graph, target, learning_rate=0.01, momentum=0.9) -> None:
        Optimizer.__init__(self, graph, target, learning_rate)

        # attenuation parameters, 0.9 is the default value
        self.momentum = momentum

        # the dict for accumulating historical speed
        self.v = dict()

    def _update(self) -> None:
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # gets the average gradient of current batch of this node
                gradient = self.get_gradient(node)

                if node not in self.v:
                    self.v[node] = - self.learning_rate * gradient
                else:
                    # moveing avergae of the gradients
                    self.v[node] = self.momentum * \
                        self.v[node] - self.learning_rate * gradient

                # update values of Variable nodes
                node.set_value(node.value + self.v[node])


class AdaGrad(Optimizer):
    """AdaGrad optimizer"""

    def __init__(self, graph, target, learning_rate=0.01) -> None:
        Optimizer.__init__(self, graph, target, learning_rate)

        # the dict for accumulating historical speed
        self.s = dict()

    def _update(self) -> None:
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # gets the average gradient of current batch of this node
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.s[node] + np.power(gradient, 2)

                node.set_value(node.value - self.learning_rate *
                               gradient / (np.sqrt(self.s[node] + 1e-10)))


class RMSProp(Optimizer):
    """RMSProp optimizer"""

    def __init__(self, graph, target, learing_rate=0.01, beta=0.9) -> None:
        Optimizer.__init__(self, graph, target, learing_rate)

        self.beta = beta

        self.s = dict()

    def _update(self) -> None:

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.beta * self.s[node] + \
                        (1 - self.beta) * np.power(gradient, 2)

                node.set_value(node.value - self.learning_rate *
                               gradient / (np.sqrt(self.s[node] + 1e-10)))


class Adam(Optimizer):
    """Adam Optimizer"""

    def __init__(self, graph, target, learning_rate=0.01, beta_1=0.9, beta_2=0.99) -> None:
        Optimizer.__init__(self, graph, target, learning_rate)

        assert 0.0 < beta_1 < 1.0
        assert 0.0 < beta_2 < 1.0
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.v = dict()
        self.s = dict()

    def _update(self) -> None:
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.v[node] = gradient
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.v[node] = self.beta_1 * self.v[node] + \
                        (1 - self.beta_1) * gradient
                    self.s[node] = self.beta_2 * self.s[node] + \
                        (1 - self.beta_2) * np.power(gradient, 2)

                node.set_value(node.value - self.learning_rate *
                               self.v[node] / np.sqrt(self.s[node] + 1e-10))
