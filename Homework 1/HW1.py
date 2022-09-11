"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 1: Programming assignment

Edited : Priyesh Rajesh Kakka
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import erdos_renyi_graph
import copy
import scipy

# -- Initialize graphs
seed = 30
G = nx.florentine_families_graph()
nodes = G.nodes()
A = nx.adjacency_matrix(G).todense()
layout = nx.spring_layout(G, seed=seed)

# -- compute jaccard's similarity

d =  list(dict(G.degree()).values())
J_num = A * A  #walk of 2

S = np.zeros(A.shape)
for i in range (len(d)):
    for j in range (len(d)):

        S[i,j] = J_num[i,j]/(d[i]+d[j] - J_num[i,j])  # Similarity matrix

# -- keep a copy of edges in the graph
old_edges = copy.deepcopy(G.edges())


new_edges, metric = [], []
new_edges_calc, metric_calc = [], []

node_names = list(G.nodes)

# -- add new edges representing similarities.
idx = -1
for u in node_names:
    idx = idx + 1
    if (u == "Ginori" ):
        idx_gio = idx
    if (u != "Ginori" ):
        new_edges_calc.append((u, "Ginori"))

metric_calc.append(S[idx_gio]) # giving values of index from similarity matrix

metric_calc = np.delete(metric_calc,idx_gio)

from pandas import *
print (DataFrame(S))

nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=600)
nx.draw_networkx_edges(G, edgelist=old_edges, pos=layout, edge_color='gray', width=4)

ne = nx.draw_networkx_edges(G, edgelist=new_edges_calc, pos=layout, edge_color=np.asarray(metric_calc), width=4, alpha=0.7, label = True)
plt.colorbar(ne)
plt.axis('off')
plt.show()
plt.savefig("From_calculation.png")

####### using networkx ##########

# pred = nx.jaccard_coefficient(G)

# for u, v, p in pred:
#     if (u == "Ginori" or v=="Ginori"):
#        # G.add_edge(u, v)
#         print(f"({u}, {v}) -> {p:.8f}")
#         new_edges.append((u, v))
#         metric.append(p)

  

# # -- plot Florentine Families graph
# nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=600)
# nx.draw_networkx_edges(G, edgelist=old_edges, pos=layout, edge_color='gray', width=4)

# # -- plot edges representing similarity

# ne = nx.draw_networkx_edges(G, edgelist=new_edges, pos=layout, edge_color=np.asarray(metric), width=4, alpha=0.7)
# plt.colorbar(ne)
# plt.axis('off')
# plt.show()
# plt.savefig("From_networkx.png")
# plt.close()


print("Finished")
