'''
Author: yitong 2969413251@qq.com
Date: 2023-02-09 19:47:56
'''
from .node import Variable
from .graph import default_graph


def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None
