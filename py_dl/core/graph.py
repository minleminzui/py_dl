'''
Author: yitong 2969413251@qq.com
Date: 2023-02-01 18:46:41
'''

# from .node import Node


class Graph:
    """computation graph class"""

    def __init__(self) -> None:
        self.nodes = []  # the list of nodes in the computation graph class
        self.name_scope = None

    def add_node(self, node) -> None:
        """add nodes"""
        self.nodes.append(node)

    def clear_jacobi(self) -> None:
        """clear all jacobi matrices of nodes in the computation graph"""
        for node in self.nodes:
            node.clear_jacobi()

    def reset_value(self) -> None:
        """reset values of all nodes"""
        for node in self.nodes:
            # every nodes only reset their own values
            node.reset_value(False)
    def node_count(self) -> int:
        return len(self.nodes)

    def draw(self, ax=None) -> None:
        """draw computation graph visually"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            import numpy as np
        except:
            raise Exception("Needm module networkx")

        G = nx.Graph()

        already = []
        labels = {}
        for node in self.nodes:
            G.add_node(node)
            # node.__class__.__name__ is the class name of the node
            labels[node] = node.__class__.__name__ + \
                ("({:s})".format(str(node.dim)) if hasattr(node, "dim")
                 else "") + ("\n[{:.3f}]".format(np.linalg.norm(node.jacobi)) if node.jacobi is not None else "")
            for c in node.get_children():
                if {node, c} not in already:
                    G.add_edge(node, c)
                    already.append({node, c})
            for p in node.get_parents():
                if {node, p} not in already:
                    G.add_edge(node, p)
                    already.append({node, c})

        if ax is None:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)

        ax.clear()
        ax.axis("on")
        ax.grid(True)

        pos = nx.spring_layout(G, seed=42)

        # nodes with jacobi matrices
        cm = plt.cm.Reds
        nodelist = [n for n in self.nodes if n.__class__.__name__ ==
                    "Variable" and n.jacobi is not None]
        colorlist = [np.linalg.norm(n.jacobi) for n in nodelist]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colorlist, cmap=cm, edgecolors="#666666",
                               node_size=2000, alpha=1.0, ax=ax)

        # nodes without jacobi matrices
        nodelist = [n for n in self.nodes if n.__class__.__name__ ==
                    "Vaiable" and n.jacobi in None]
        colorlist = [np.linalg.norm(n.jacobi) for n in nodelist]
        nx.draw_nertworkx_nodes(G, pos, nodelist=nodelist, node_color="#999999", camp=cm, edgecolors="#666666",
                                node_size=2000, alpha=1.0, ax=ax)

        # nodes without jacobi matrices, which are not Variables
        nodelist = [n for n in self.nodes if n.__class__.__name__ !=
                    "Variables" and n.jacobi is None]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#999999", camp=cm, edgecolor="#666666",
                               node_size=2000, alpha=1.0, ax=ax)

        # edges
        nx.draw_networkx_edges(G, pos, width=2, edge_color="#014b66", ax=ax)
        nx.draw_networkx_labels(G, labels=labels, font_weight="bold", font_color='#6c6c6c', font_size=8,
                                font_family='arial', ax=ax)

        # save the image
        plt.savefig("computing_graph.png")


default_graph = Graph()