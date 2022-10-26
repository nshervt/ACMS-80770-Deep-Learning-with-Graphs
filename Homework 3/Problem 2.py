"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 2: Programming assignment
Problem 2
"""
import torch

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch import nn
from torch.autograd.functional import jacobian


torch.manual_seed(0)


class GCN:
    """
        Graph convolutional layer
    """
    def __init__(self, in_features, out_features):
        # -- initialize weight
        pass

        # -- non-linearity

    def __call__(self, A, H):
        # -- GCN propagation rule
        pass


class MyModel(nn.Module):
    """
        model
    """
    def __init__(self, A):
        super(MyModel, self).__init__()
        # -- initialize layers
        pass

        self.A = A

    def forward(self, h0):
        pass


"""
    Effective range
"""
# -- Initialize graph
seed = 32
n_V = 200   # total number of nodes
i = 17      # node ID
k = 0       # k-hop
G = nx.barabasi_albert_graph(n_V, 2, seed=seed)

# -- plot graph
layout = nx.spring_layout(G, seed=seed, iterations=400)
nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)

# -- plot neighborhood
nodes = nx.single_source_shortest_path_length(G, i, cutoff=k)
im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

# -- visualize
plt.colorbar(im2)
plt.show()
plt.close()


"""
    Influence score
"""
# -- Initialize the model and node feature vectors
# model = ?
# H = ?

# -- Influence sore
# inf_score = ?

# -- plot influence scores
