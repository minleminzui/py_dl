'''
Author: yitong 2969413251@qq.com
Date: 2023-02-23 18:00:52
'''
import json
import os
import datetime

import numpy as np

from ..core.core import get_node_from_graph
from ..core import *
from ..core import Node, Variable
from ..core.graph import default_graph
from ..ops import *
from ..ops.loss import *
from ..ops.metrics import *
# from ..util import


class Saver(object):
    """models, computation graph storer and loader class
    models are stored as two separate files:
    1. the structure meta information of the computation graph
    2. the values of nodes in the computations. More specifically, the weights of variable nodes 
    """

    def __init__(self, root_dir=''):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def save(self, graph=None, meta=None, service_signature=None,
             model_file_name='model.json',
             weights_file_name='weights.npz'):
        """save the computation graph into files"""

        if graph is None:
            graph = default_graph

        # meta message, mainly record the time for saving models and file names of nodes' name
        meta = {} if meta is None else meta
        meta['save_time'] = str(datetime.datetime.now())
        meta['weights_file_name'] = weights_file_name

        # descriptions of the service interface
        service = {} if service_signature is None else service_signature

        # start saving
        self._save_model_and_weights(
            graph, meta, service, model_file_name, weights_file_name)

    def _save_model_and_weights(self, graph, meta, service, model_file_name, weights_file_name):
        model_json = {
            'meta': meta,
            'service': service
        }
        graph_json = []
        weights_dict = dict()

        # save node meta information as dict/json
        for node in graph.nodes:
            if not node.need_save:
                continue
            # node.kargs.pop('name', None)
            node_json = {
                'node_type': node.__class__.__name__,
                'name': node.name,
                'parents': [parent.name for parent in node.parents],
                'children': [child.name for child in node.children],
                'kargs': node.kargs
            }

            # save the dim of nodes
            if node.value is not None:
                if isinstance(node.value, np.matrix):
                    node_json['dim'] = node.value.shape

            # if nodes are Variable type, save its value
            # we don't need to save other types' value
            if isinstance(node, Variable):
                weights_dict[node.name] = node.value

        model_json['graph'] = graph_json

        # save computation meta information as json
        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'w') as model_file:
            json.dump(model_json, model_file, indent=4)
            print(f'Save model into file: {model_file.name}')

        # save values of nodes as npz (only for Variable)
        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'wb') as weights_file:
            np.savez(weights_file, **weights_dict)
            print(f'Save weights to file: {weights_file.name}')

    @staticmethod
    def create_node(graph, from_model_json, node_json):
        """static until function, construct non-exist nodes recrusively"""
        node_type = node_json['node_type']
        node_name = node_json['name']
        parents_name = node_json['parents']
        dim = node_json.get('dim', None)
        kargs = node_json.get('kargs', None)
        kargs['graph'] = graph

        parents = []
        for parent_name in parents_name:
            parent_node = get_node_from_graph(parent_name, graph=graph)
            if parent_node is None:
                parent_node_json = None
                for node in from_model_json:
                    if node['name'] == parent_name:
                        parent_node_json = node
                assert parent_node_json is not None
                # if there is not a parent node, call recursively
                parent_node = Saver.create_node(
                    graph, from_model_json, parent_node_json)
            parents.appen(parent_node)

        # reflection creates a node instance
        if node_type == 'Variable':
            assert dim is not None
            dim = tuple(dim)
            return Class

    def load(self, to_graph=None, model_file_name='model.json', weights_file_name='weights.npz'):
        """read from files and restore the corresponding computation graph"""

        if to_graph is None:
            to_graph = default_graph

        model_json = {}
        graph_json = []
        weights_dict = dict()

        # read computation graph meta data
        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'r') as model_file:
            model_json = json.load(model_file)

        # read values of computation graph Variable nodes
        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'rb') as weights_file:
            weights_npz_files = np.load(weights_file)
            for file_name in weights_npz_files.files:
                weights_dict[file_name] = weights_npz_files[file_name]
            weights_npz_files.close()

        graph_json = model_json['graph']
        self._restore_nodes(to_graph, graph_json, weights_dict)
