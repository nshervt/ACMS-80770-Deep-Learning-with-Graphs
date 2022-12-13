# import packages torch, numpy, networks etc...
from torch_geometric.datasets import QM9
import matplotlib.pyplot as plt
import sklearn
import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import random
from typing import List
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec

# pick up train_datasets and test_datasets from the total QM9 datasets
dataset = QM9('data')
train_data = dataset[:3000]
test_data = dataset[10001:10501]

# the data in the dataset is trained one by one cause I have not figure out how to put batch data into deepwalk method, so the next 
# two lines are not necessary
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

# define the Deepwalk method to get node embeddings 
class DeepWalk:
    def __init__(self, window_size: int, embedding_size: int, walk_length: int, walks_per_node: int):
        """
        :param window_size: window size for the Word2Vec model
        :param embedding_size: size of the final embedding
        :param walk_length: length of the walk
        :param walks_per_node: number of walks per node
        """
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.walk_length = walk_length
        self.walk_per_node = walks_per_node

    def random_walk(self, g: nx.Graph, start: str, use_probabilities: bool = False) -> List[str]:
        """
        Generate a random walk starting on start
        :param g: Graph
        :param start: starting node for the random walk
        :param use_probabilities: if True take into account the weights assigned to each edge to select the next candidate
        :return:
        """
        walk = [start]
        for i in range(self.walk_length):
            neighbours = g.neighbors(walk[i])
            neighs = list(neighbours)
            if use_probabilities:
                probabilities = [g.get_edge_data(walk[i], neig)["weight"] for neig in neighs]
                sum_probabilities = sum(probabilities)
                probabilities = list(map(lambda t: t / sum_probabilities, probabilities))
                p = np.random.choice(neighs, p=probabilities)
            else:
                p = random.choice(neighs)
            walk.append(p)
        return walk

    def get_walks(self, g: nx.Graph, use_probabilities: bool = False) -> List[List[str]]:
        """
        Generate all the random walks
        :param g: Graph
        :param use_probabilities:
        :return:
        """
        random_walks = []
        for _ in range(self.walk_per_node):
            random_nodes = list(g.nodes)
            random.shuffle(random_nodes)
            for node in tqdm(random_nodes):
                random_walks.append(self.random_walk(g=g, start=node, use_probabilities=use_probabilities))
        return random_walks

    def compute_embeddings(self, walks: List[List[str]]):
        """
        Compute the node embeddings for the generated walks
        :param walks: List of walks
        :return:
        """
        model = Word2Vec(sentences=walks, window=self.window_size, vector_size=self.embedding_size)
        return model.wv

# define a linear model to predict y value (molecular properties)
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# define train criterion and optimizer
lmodel = Model(64,1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(lmodel.parameters(), lr = 0.01)

# train the train_dataset: first get the node embeddings, use average of node embedding to represent the graph embedding
# and put the graph embedding into the linear regrassion model defined above
epochs = 1 
losses = []
import time
start_time = time.time()

for i in range(epochs):
    for j in range(len(train_data)):
        G = to_networkx(train_data[j], node_attrs=["x"], edge_attrs=["edge_attr"]) # transfer it into a networkx graph
        deepwalk = DeepWalk(2,64,3,1) # set window size, embedding size, walk length and walk per node
        walks = deepwalk.get_walks(G) # get walks 
        emb = deepwalk.compute_embeddings(walks) # compute node embedding
        emb = torch.from_numpy(np.mean(emb.get_normed_vectors(),axis=0)) # average the node embedding to get graph embedding
        y_pred = lmodel.forward(emb) # predict y value
        y=train_data[j].y[:,1] # Isotropic polarizability of molecules, could be other physical or chemical properties
        loss = criterion(y_pred,y) # RMSE
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        j+=1

# plot loss every 50 graphs, uncomment 9 line below to see loss function 
# losses1 = []
# for i in range(len(losses)):
#     if i%50 ==1:
#         losses1.append(losses[i])

# with torch.no_grad():
#     plt.plot(range(len(losses1)), losses1)
#     plt.ylabel('Loss')
#     plt.xlabel('num_every_50_graphs')

# predict y value for test_dataset
y_vals = []
y_tests = []
with torch.no_grad():
    for data in test_data:
        G = to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"])
        deepwalk = DeepWalk(2,64,3,1)
        walks = deepwalk.get_walks(G)
        emb = deepwalk.compute_embeddings(walks)
        emb = torch.from_numpy(np.mean(emb.get_normed_vectors(),axis=0))
        y_pred = lmodel.forward(emb)
        y_vals.append(y_pred)
        y_test = data.y[:,1]
        y_tests.append(y_test)
        loss = torch.sqrt(criterion(y_pred, y_test))

# print first the y value of first 50 data in test_dataset     
print(f'{"PREDICTED":>12} {"ACTUAL":>8} {"DIFF":>8}')
diffs=[]
errors=[]
for i in range(50):
    diff = np.abs(y_vals[i].item()-y_tests[i].item())
    diffs.append(diff)
    error = diff/y_tests[i].item()
    errors.append(error)
    print(f'{i+1:2}. {y_vals[i].item():8.4f} {y_tests[i].item():8.4f} {diff:8.4f}')

# plot the predict and actual isotropic polarizability of first 50 molecules 
plt.plot(range(50), y_vals[:50], label = "predictions")
plt.plot(range(50), y_tests[:50], label = "real values")
ax = plt.gca()
ax.set_ylim([0, 100])
plt.ylabel('Isotropic polarizability')
plt.xlabel('Molecules')
plt.legend()
plt.show()