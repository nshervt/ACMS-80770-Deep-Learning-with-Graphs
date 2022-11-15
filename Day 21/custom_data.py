"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 2: Programming assignment
Problem 1
"""
import torch
from torch import nn
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=UserWarning)
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor
# from torchvision import datasets
from torch.utils.data import Dataset

"""
    load data
"""
class MolecularDataset(Dataset):
    def __init__(self):
        dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True),
                                                   return_smiles=True,
                                                   target_index=np.random.choice(range(133000), 100, False))

        self.atom_types = [6, 8, 7, 9, 1]
        self.V = 9

        self.adjs = torch.stack(list(map(self.adj, dataset)))
        self.sigs = torch.stack(list(map(self.sig, dataset)))
        self.prop = torch.stack(list(map(self.target, dataset)))[:, 5]

    def adj(self, x):
        x = x[1]
        adjacency = np.zeros((self.V, self.V)).astype(float)
        adjacency[:len(x[0]), :len(x[0])] = x[0] + 2 * x[1] + 3 * x[2]
        return torch.tensor(adjacency)

    def sig(self, x):
        x = x[0]
        atoms = np.ones((self.V)).astype(float)
        atoms[:len(x)] = x
        out = np.array([int(atom == atom_type) for atom_type in self.atom_types for atom in atoms]).astype(float)
        return torch.tensor(out).reshape(5, len(atoms)).T

    def target(self, x):
        x = x[2]
        return torch.tensor(x)

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, item):
        return self.adjs[item], self.sigs[item], self.prop[item]


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


class GraphPooling:
    """
        Graph pooling layer
    """
    def __init__(self):
        pass

    def __call__(self, H):
        # -- multi-set pooling operator
        pass


class MyModel(nn.Module):
    """
        Regression model
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # -- initialize layers
        pass

    def forward(self, A, h0):
        pass

quit()
"""
    Train
"""
# -- Initialize the model, loss function, and the optimizer
model = MyModel()
# MyLoss =
# MyOptimizer =

# -- update parameters
for epoch in range(200):
    for i in range(10):

        # -- predict
        pred = model(adjs[i*10:(i+1)*10], sigs[i*10:(i+1)*10])

        # -- loss
        # loss = ?

        # -- optimize

# -- plot loss
