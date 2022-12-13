"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 1: Programming assignment
"""

from pickletools import float8
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import erdos_renyi_graph
import copy


# -- Initialize graphs
seed = 30
G = nx.florentine_families_graph()
nodes = G.nodes()

layout = nx.spring_layout(G, seed=seed)

# -- compute jaccard's similarity
"""
    This example is using NetwrokX's native implementation to compute similarities.
    Write a code to compute Jaccard's similarity and replace with this function.
"""
pred = nx.jaccard_coefficient(G)

A = nx.adjacency_matrix(G)
S = np.zeros_like(A.todense(), dtype=float)
for i in range(A.shape[0]):
    for j in range(A.shape[0]):
        S[i,j] = np.array((sum(A[:,i].multiply(A[:,j])) 
                / len((A[:,i]+A[:,j]).nonzero()[0])).todense())[0][0]

Ginori_dict = {}
for i, nodei in enumerate(nodes):
    for j, nodej in enumerate(nodes):
        if nodei == 'Ginori':
         Ginori_dict[nodej] = S[i,j]


# -- keep a copy of edges in the graph
old_edges = copy.deepcopy(G.edges())

# -- add new edges representing similarities.
"""
    This is an example to show how to add edges to a graph. You may need to modify the 
    loop and don’t need to use the loop as it is.
"""
# new_edges, metric = [], []
# for u, v, p in pred:
#     G.add_edge(u, v)
#     print(f"({u}, {v}) -> {p:.8f}")
#     new_edges.append((u, v))
#     metric.append(p)

new_edges, metric = [], []
for v in Ginori_dict:
    u = 'Ginori'
    p = Ginori_dict[v]
    G.add_edge(u, v)
    print(f"({u}, {v}) -> {p:.8f}")
    new_edges.append((u, v))
    metric.append(p)

# # -- plot Florentine Families graph
# nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=600)
# nx.draw_networkx_edges(G, edgelist=old_edges, pos=layout, edge_color='gray', width=4)

# # -- plot edges representing similarity
# """
#     This example is randomly plotting similarities between 8 pairs of nodes in the graph. 
#     Identify the ”Ginori”
# """
# ne = nx.draw_networkx_edges(G, edgelist=new_edges[:8], pos=layout, edge_color=np.asarray(metric[:8]), width=4, alpha=0.7)
# plt.colorbar(ne)
# plt.axis('off')
# plt.show()


# -- plot Florentine Families graph
nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=600)
nx.draw_networkx_edges(G, edgelist=new_edges, pos=layout, edge_color='gray', width=4)

# -- plot edges representing similarity
"""
    This example is randomly plotting similarities between 8 pairs of nodes in the graph. 
    Identify the ”Ginori”
"""
ne = nx.draw_networkx_edges(G, edgelist=new_edges, pos=layout, edge_color=np.asarray(metric), width=4, alpha=0.7)
plt.colorbar(ne)
plt.axis('off')
plt.savefig('./similarity.png')
