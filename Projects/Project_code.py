# Deep Learning with Graphs
# Course project: Coarsening graph networks which arise in the optimal design of robotic mechanisms
# Author: Aravind Baskar

# Import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

# Initialize the original graph and the seed for initial layout
# Example 1
source = [0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
          14, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20]
target = [1, 2, 0, 3, 0, 1, 4, 3, 5, 7, 8, 9, 4, 6, 5, 7, 8, 4, 6, 4, 6, 4, 10, 9, 11, 10, 12, 11, 16, 14, 15, 13, 16,
          17, 13, 12, 14, 14, 18, 17, 19, 18, 20, 19]

# Example 2
# source = [0, 1, 2, 3, 4, 5, 6, 0, 3, 8, 4, 10, 1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11]
# target = [1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 0, 3, 8, 4, 10]

# Example 3
# source = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 6]
# target = [1, 2, 3, 4, 5, 0, 6, 0, 1, 2, 3, 4, 5, 0]

seed = 9

# Construct the adjacency matrix from the edge lists
m = max(max(source), max(target)) + 1
v = [1 for k in range(len(source))]
A = sparse.csr_matrix((v, (source, target)), shape=(m, m)).toarray()

# Initialize the graph and layout
G = nx.from_numpy_matrix(A)
layout = nx.spring_layout(G, seed=seed)

# Iterative coarsening of the graph via spectral grouping based on eigenvector centrality measure.

Ak = A
max_iter = 10
count = 0

while count < max_iter:  # Set an iteration limit to avoid indefinite loop
    print("Initiate iteration: " + str(count + 1))
    # Visualize the graph at the current iteration based on eigenvector centrality measure
    Gk = nx.from_numpy_matrix(Ak)
    nodes = Gk.nodes()
    edges = Gk.edges()

    print(Ak)

    # Eigen-centrality computation using power method

    # Define a starting vector
    x = np.ones(len(Ak))

    # Define the convergence threshold
    threshold = 1e-6

    num = 0
    # Iterate until convergence
    while num < 1000:
        # Compute the new vector
        x_new = (Ak + 5 * np.eye(len(Ak))) @ x

        # Normalize the vector
        x_new = x_new / np.linalg.norm(x_new)

        # Compute the relative change in the vector
        change = np.abs((x_new - x) / x_new)

        # Check if the vector has converged
        if change.max() < threshold:
            print("Power method converged after " + str(num + 1) + " trial(s)")
            break

        # Update the vector
        x = x_new

        num += 1

    score = x

    print(score)

    # Graph visualization

    plt.figure(count, figsize=(20, 20))
    plt.title("Graph coarsening by spectral grouping, Iteration: " + str(count))
    nx.draw_networkx_labels(Gk, pos=layout, font_size=12, font_color='white', font_family='sans-serif')
    nx.draw_networkx_nodes(Gk, nodelist=nodes, label=nodes, node_color=score, pos=layout, node_size=500)
    nx.draw_networkx_edges(Gk, edgelist=edges, pos=layout, edge_color='magenta', width=4)
    plt.axis([-1.05, 1.05, -1.05, 1.05])
    plt.show()

    # Termination criteria is when the graph becomes a lone node
    if len(Ak) == 1:
        print("Coarsening terminated upon reaching a lone node")
        break

    # Coarsening grouping is done via spectral grouping following Webb et al.
    sort_score = np.argsort(score)

    # Spectral grouping is based on the eigenvector centrality
    ind = np.zeros(len(nodes))
    roll = 0
    counter = 0
    check = 0
    n_weights = []
    for a in sort_score:
        nlist = [b for b in Gk.neighbors(a)]
        n_score = min([score[b] for b in nlist])
        if ind[a] == 0:
            if counter != 0:
                n_weights.append(counter)
            roll += 1
            counter = 0
            ind[a] = roll
            counter += 1
            print("Assign type-A: " + str(a) + " Group: " + str(roll - 1))
        else:
            continue
        tf_cond = True
        for c in nlist:
            if (ind[c] == 0) and (0 <= score[c] - n_score < 0.15):
                if tf_cond:
                    ind[c] = roll
                    counter += 1
                    check = score[c]
                    tf_cond = False
                    print("Assign type-B: " + str(c) + " Group: " + str(roll - 1))
                else:
                    if score[c] <= check:
                        ind[c] = roll
                        counter += 1
                    else:
                        print("Assign type-B by-passed")
                        continue
                clist = [b for b in Gk.neighbors(c)]
                for d in clist:
                    if (ind[d] == 0) and (abs(score[d] - score[a]) < 10 ** -4):
                        print("Assign type-C: " + str(d) + " Group: " + str(roll - 1))
                        ind[d] = roll
                        counter += 1
    if len(n_weights) < roll:
        n_weights.append(counter)

    ind = [int(k - 1) for k in ind]

    n_weights = [int(k - 1) for k in n_weights]
    # n_weights = n_weights/np.linalg.norm(n_weights)

    print([[i, j] for i, j in zip(ind, score)])

    # Update the adjacency matrix for next iteration
    Source = [ind[k] for k in source]
    Target = [ind[k] for k in target]
    V = [1 for k in range(len(Source))]
    Ak = sparse.csr_matrix((V, (Source, Target)), shape=(roll, roll)).toarray()
    Ak = np.where(Ak > 0, 1, 0)
    Ak = Ak - np.diag(np.diag(Ak)) + np.diag(n_weights)
    source = Source
    target = Target

    # Earlier layout is modified following a weighted centroid rule.
    centroids = []
    for k in range(roll):
        t = [j == k for j in ind]
        centroids.append(sum([score[i] * layout[i] for i, x in enumerate(t) if x]) /
                         sum([score[i] for i, x in enumerate(t) if x]))
        # Replace score[i] by 1 in the above for simple centroid rule

    layout = dict(zip([j for j in range(roll)], centroids))
    count += 1
