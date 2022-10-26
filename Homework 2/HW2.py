"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 2: Programming assignment
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import erdos_renyi_graph
import copy


# -- Initialize graphs
seed = 32
G = nx.karate_club_graph()
nodes = G.nodes()
edges = G.edges()

layout = nx.spring_layout(G, seed=seed)

# -- find Laplacian
L = nx.laplacian_matrix(G).todense()
L = nx.normalized_laplacian_matrix(G).todense()

D = np.diag([val**(-0.5) for (node, val) in G.degree()])

L_sym = np.matmul(np.matmul(D, L), D)
print(L_sym)



print(L_sym.sum(axis=1).P)

print(nx.normalized_laplacian_matrix(G).todense().sum(axis=1))

Lamb, V = np.linalg.eigh(L)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=2)


X = np.asarray(V.P)


# print(Lamb, X, L)

print(L.sum(axis=1))

quit()

# print(X[0])
# print(X[1])




from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

X = normalize(X, axis=1, norm='l2')

kmeans = KMeans(n_clusters=2, random_state=0, n_init=200).fit(X[0:2].P)

print(kmeans.labels_)
color_vec = []

for label in kmeans.labels_:
    if label == 1:
        color_vec.append('red')
    elif label == 0:
        color_vec.append('blue')


# -- plot karate club graph
nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=300, node_color=color_vec)
nx.draw_networkx_edges(G, edgelist=edges, pos=layout, edge_color='gray', width=3)

plt.axis('off')
plt.show()
plt.close()

plt.scatter(X[0], X[1], c=color_vec)
plt.show()
plt.close()


