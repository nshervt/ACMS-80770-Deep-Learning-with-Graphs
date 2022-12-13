"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 2: Programming assignment
Problem 2
"""
#%%
import torch

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch import nn
from torch.autograd.functional import jacobian


torch.manual_seed(0)

device = torch.device("cpu")


class GCN:
    """
        Graph convolutional layer
    """
    def __init__(self, in_features, out_features):
        # -- initialize weight
        
        self.W = torch.nn.Parameter(torch.rand((in_features, out_features)), requires_grad=True).type(torch.DoubleTensor).to(device)
        
        # -- non-linearity

    def __call__(self, A, H,node):
        
        # -- GCN propagation rule
        I = torch.eye(A.shape[1]).to(device)
       
        AI = A + torch.eye(A.shape[1]).to(device)
        D = I*torch.sum(AI,1)
        Dinv_hf = torch.sqrt(torch.inverse(D))

        Hn = torch.mm(torch.mm(torch.mm(torch.mm(Dinv_hf,AI),Dinv_hf),H),self.W)
        if node: Hn = Hn[node,:]
        
        return (Hn)
        



   
"""
    Effective range
"""
# -- Initialize graph
seed = 32
n_V = 200   # total number of nodes
  
nodeids = [17,27] # node ID
khops   = [2,4,6] # k-hop

G = nx.barabasi_albert_graph(n_V, 2, seed=seed)

## Intializing node attributes
H = torch.eye(200,requires_grad=True).type(torch.DoubleTensor)

# -- Initialize the model and node feature vectors
A = torch.from_numpy(nx.to_numpy_matrix(G))

# -- plot neighborhood
for i in nodeids:
    for k in khops:
        
        G = nx.barabasi_albert_graph(n_V, 2, seed=seed)
        layout = nx.spring_layout(G, seed=seed, iterations=400)
        
        nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
        nodes = nx.single_source_shortest_path_length(G, i, cutoff=k)
        
        im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

        plt.savefig("./Homework3/effective_range_node"+str(i)+"_layers"+str(k))
        plt.close()


"""
    Influence score
"""


##### For Model 1 200-100 #############################################
for node in nodeids:

    class MyModel(nn.Module):
        """
            Regression model
        """
        def __init__(self,A):
            super(MyModel, self).__init__()
            # -- initialize layers
            self.GNN = GCN(in_features = 200 , out_features = 100)
            self.A = A
            
        def forward(self, h0,node):
            return  self.GNN(self.A,h0,node)
            
    model = MyModel(torch.tensor(A,requires_grad=True).type(torch.DoubleTensor))


    Hk = lambda H :  model((H),node)
    J = torch.autograd.functional.jacobian(Hk,(H))

    # -- Influence score
    Ik=torch.empty(J.shape[1],1)
    for i in range (len(J[1])):
        Ik[i] = torch.mm(torch.mm(torch.ones(1,J.shape[0]).type(torch.DoubleTensor),J[:,i,:]), torch.ones(J.shape[2],1).type(torch.DoubleTensor))
    

    G = nx.barabasi_albert_graph(n_V, 2, seed=seed)
    layout = nx.spring_layout(G, seed=seed, iterations=400)

    nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
    
    vmin_const = min(Ik)
    vmax_const = max(Ik)
    
    im2 = nx.draw_networkx_nodes(G, node_color=(Ik), pos=layout, node_size=100,vmin=vmin_const,vmax=vmax_const) ;plt.colorbar(im2,shrink=0.8)

    plt.savefig("./Homework3/Model1_node"+str(node))
    plt.close()

##### For Model 2 200-100-50-20 #############################################
for node in nodeids:

    class MyModel(nn.Module):
        """
            Regression model
        """
        def __init__(self,A):
            super(MyModel, self).__init__()
            # -- initialize layers
            self.GNN1 = GCN(in_features = 200 , out_features = 100)
            self.GNN2 = GCN(in_features = 100 , out_features = 50)
            self.GNN3 = GCN(in_features = 50  , out_features = 20)
            self.A = A
            
        def forward(self, h0,node):
            return self.GNN3(self.A,self.GNN2(self.A,self.GNN1(self.A,h0, None),None),node)
            
            
    model = MyModel(torch.tensor(A,requires_grad=True).type(torch.DoubleTensor))


    Hk = lambda H :  model((H),node)
    J = torch.autograd.functional.jacobian(Hk,(H))

    # -- Influence score
    Ik=torch.empty(J.shape[1],1)
    for i in range (len(J[1])):
        Ik[i] = torch.mm(torch.mm(torch.ones(1,J.shape[0]).type(torch.DoubleTensor),J[:,i,:]), torch.ones(J.shape[2],1).type(torch.DoubleTensor))
    

    G = nx.barabasi_albert_graph(n_V, 2, seed=seed)
    layout = nx.spring_layout(G, seed=seed, iterations=400)

    nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)


    im2 = nx.draw_networkx_nodes(G, node_color=(Ik), pos=layout, node_size=100,vmin=vmin_const,vmax=vmax_const) ;plt.colorbar(im2,shrink=0.8)

    plt.savefig("./Homework3/Model2_node"+str(node))
    plt.close()

##### For Model 3 200-100-50-20-20-20 #############################################
for node in nodeids:

    class MyModel(nn.Module):
        """
            Regression model
        """
        def __init__(self,A):
            super(MyModel, self).__init__()
            # -- initialize layers
            self.GNN1 = GCN(in_features = 200 , out_features = 100)
            self.GNN2 = GCN(in_features = 100 , out_features = 50)
            self.GNN3 = GCN(in_features = 50  , out_features = 20)
            self.GNN4 = GCN(in_features = 20  , out_features = 20)
            self.A = A
            
        def forward(self, h0,node):
            return self.GNN4(self.A,self.GNN3(self.A,self.GNN2(self.A,self.GNN1(self.A,h0,None), None), None),node)
            
            
    model = MyModel(torch.tensor(A,requires_grad=True).type(torch.DoubleTensor))


    Hk = lambda H :  model((H),node)
    J = torch.autograd.functional.jacobian(Hk,(H))

    # -- Influence score
    Ik=torch.empty(J.shape[1],1)
    for i in range (len(J[1])):
        Ik[i] = torch.mm(torch.mm(torch.ones(1,J.shape[0]).type(torch.DoubleTensor),J[:,i,:]), torch.ones(J.shape[2],1).type(torch.DoubleTensor))
    

    G = nx.barabasi_albert_graph(n_V, 2, seed=seed)
    layout = nx.spring_layout(G, seed=seed, iterations=400)

    nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)


    im2 = nx.draw_networkx_nodes(G, node_color=(Ik), pos=layout, node_size=100,vmin=vmin_const,vmax=vmax_const) ;plt.colorbar(im2,shrink=0.8)

    plt.savefig("./Homework3/Model3_node"+str(node))
    plt.close()

