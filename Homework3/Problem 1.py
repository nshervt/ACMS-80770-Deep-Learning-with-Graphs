"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 2: Programming assignment
Problem 1
"""

import time
import torch
from torch import nn
import warnings
import numpy as np

# warnings.simplefilter(action='ignore', category=UserWarning)
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor

import matplotlib.pyplot as plt

"""
    load data
"""
dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True), return_smiles=True,
                                           target_index=np.random.choice(range(133000), 6000, False))

device = torch.device('cpu')
V = 9
atom_types = [6, 8, 7, 9, 1]

def adj(x):
    x = x[1]
    adjacency = np.zeros((V, V)).astype(float)
    adjacency[:len(x[0]), :len(x[0])] = x[0] + 2 * x[1] + 3 * x[2]
    return torch.tensor(adjacency)


def sig(x):
    x = x[0]
    atoms = np.ones((V)).astype(float)
    atoms[:len(x)] = x
    out = np.array([int(atom == atom_type) for atom_type in atom_types for atom in atoms]).astype(float)
    return torch.tensor(out).reshape(5, len(atoms)).T


def target(x):
    x = x[2]
    return torch.tensor(x)


adjs = torch.stack(list(map(adj, dataset))).to(device)
sigs = torch.stack(list(map(sig, dataset))).to(device)
prop = torch.stack(list(map(target, dataset)))[:, 5].to(device)



class GCN(nn.Module):
    """
        Graph convolutional layer
    """
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        # -- initialize weight
        self.W = nn.Parameter(torch.rand((in_features, out_features))*0.01, requires_grad=True).type(torch.DoubleTensor).to(device)

        # -- non-linearity

    def __call__(self, A, H):
        # -- GCN propagation rule
        I = torch.eye(A.shape[1]).to(device)
        Hn = []
        for i in range(len(A)):
            AI      = A[i]+I
            D       = I*torch.sum(AI,0)
            Dinv_hf = torch.sqrt(torch.inverse(D))
            DAD     = torch.mm(torch.mm(Dinv_hf,AI),Dinv_hf)
            HW      = torch.mm(H[i], self.W)
            Hn.append(torch.unsqueeze(torch.mm(DAD, HW), 0))
        return torch.cat((Hn))

        


class GraphPooling:
    """
        Graph pooling layer
    """
    def __init__(self):
        pass

    def __call__(self, H):
        # -- multi-set pooling operator
        return torch.sum(H, 1).type(torch.DoubleTensor).to(device)


class MyModel(nn.Module):
    """
        Regression  model
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # -- initialize layers
        hidd_features = 3
        self.gcn_layer = GCN(in_features = sigs[0].shape[1], out_features = hidd_features)
        self.Gpool = GraphPooling()
        # self.fc = nn.Sequential(nn.Linear(6,1),nn.LeakyReLU())
        self.fc = nn.Linear(hidd_features,1)
        self.relu = nn.ReLU()
    def forward(self, A, h0):
        h1 = self.gcn_layer(A,h0)
        h1 = self.relu(h1)
        h2 = self.Gpool(h1)
        h3 = self.fc(h2.type(torch.float))
        return h3



"""
    Train
"""
# -- Initialize the model, loss function, and the optimizer
model = MyModel().to(device)
MyLoss = nn.MSELoss().to(device)
MyOptimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
error = []


training_points = 5000
testing_pints = 1000
batch = 10
dp = training_points//batch

# -- update parameters
for epoch in range(200):
    loss_sum = 0
    # print("epoch ",epoch)
    for i in range(batch):
        
        # -- predict
        pred = model(adjs[i*dp:(i+1)*dp], sigs[i*dp:(i+1)*dp])
        targ = prop[i*dp:(i+1)*dp][:,None]

        # -- loss
        loss = MyLoss(pred, targ)
        loss.backward()
        
        # -- optimize
        MyOptimizer.step()
        MyOptimizer.zero_grad()
        
        
        loss_sum += loss.item()
    
    print("epoch ",epoch, loss_sum)
    error.append(loss_sum)
    
# -- plot loss
error = np.array(error)


plt.plot(error)
plt.savefig('./Homework3/training_error.png')
plt.close()

"""
Test
"""

pred = model(adjs[training_points:training_points+testing_pints], sigs[training_points:training_points+testing_pints]).to('cpu').detach().numpy()
targ = prop[training_points:training_points+testing_pints].to('cpu')

plt.close('all')
plt.plot(pred[:,0], targ, '^')
x = torch.linspace(torch.min(targ),torch.max(targ),100)
plt.plot(x,x)
plt.xlim(min(pred[:,0]),max(pred[:,0]))
plt.savefig('./Homework3/evaluvation.png')

