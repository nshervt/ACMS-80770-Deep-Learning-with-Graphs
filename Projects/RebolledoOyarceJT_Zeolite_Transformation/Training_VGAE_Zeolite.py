#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv

# Reading csv file into numpy array 
Adjs_Matrix_File    = np.loadtxt("data_features/data_AdjsFWs.csv", delimiter=",")
SOAP_Matrix_File    = np.loadtxt("data_features/data_TensorFWs.csv", delimiter=",")

Name_FWs = np.array([])
NAtoms_FWs = np.array([])
with open("data_features/data_NameFW_NAtoms.csv", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        Name_FWs = np.append(Name_FWs, row[0])
        NAtoms_FWs = np.append(NAtoms_FWs, int(row[1]))
# printing the array

print(Adjs_Matrix_File.shape, SOAP_Matrix_File.shape, Name_FWs.shape, NAtoms_FWs.shape)


# In[2]:


n,m = Adjs_Matrix_File.shape

for i in range(n):
    arr_current_reshape = Adjs_Matrix_File[i].reshape((int(np.sqrt(m)), int(np.sqrt(m))))

    if i == 0:
        Adjs_FWs_Tensor = arr_current_reshape
    elif i == 1:
        Adjs_FWs_Tensor = np.array([Adjs_FWs_Tensor, arr_current_reshape])
    else:
        Adjs_FWs_Tensor = np.append(Adjs_FWs_Tensor, [arr_current_reshape], axis = 0)

print(Adjs_FWs_Tensor.shape)


# In[3]:


n,m_F = SOAP_Matrix_File.shape

for i in range(n):
    arr_current_reshape = SOAP_Matrix_File[i].reshape((int(np.sqrt(m)), int(m_F/np.sqrt(m))))

    if i == 0:
        Features_Tensor = arr_current_reshape
    elif i == 1:
        Features_Tensor = np.array([Features_Tensor, arr_current_reshape])
    else:
        Features_Tensor = np.append(Features_Tensor, [arr_current_reshape], axis = 0)

print(Features_Tensor.shape)


# In[4]:


import torch

Adjs_FWs_Tensor = torch.tensor(Adjs_FWs_Tensor, dtype = torch.float32)
Features_Tensor = torch.tensor(Features_Tensor, dtype = torch.float32)


# In[5]:


print(Features_Tensor.shape)
print(Adjs_FWs_Tensor.shape)


# In[6]:



# In[7]:


from utils import *
import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from VGAE_Model import *


def Training_VGAE(AdjacencyMatrix, FeaturesTensor, N_Atoms, n_epochs, batch_size, lr):
    
    FeaturesTensor_Normalize = torch.nn.functional.normalize(FeaturesTensor)
    dim_outputlayer = FeaturesTensor.shape[1]
    dim_inputlayer  = FeaturesTensor.shape[2]
    
    
    n_graphs = AdjacencyMatrix.shape[0]
    Adj_hat = Get_AdjHat(AdjacencyMatrix)
    Adj_hat_reshape = torch.reshape(Adj_hat,(n_graphs,-1))
    
    
    # -- Initialize the model, loss function, and the optimizer
    model = MyModel(dim_inputlayer, dim_outputlayer)

    MyLoss = nn.BCELoss()

    MyOptimizer = optim.Adam(model.parameters(), lr=lr)
    loss_epoch = []


    for epoch in range(1,n_epochs+1):

        cum_loss = 0
        
        for i in range(10):

            MyOptimizer.zero_grad()
            
            if i == 9:
             # -- predict
                pred_reshape, pred, mean, logstd, lat_variable  = model(AdjacencyMatrix[i*batch_size:], FeaturesTensor_Normalize[i*batch_size:])

                current_Natoms = N_Atoms[i*batch_size:]
            
                labels = Adj_hat_reshape[i*batch_size:]   # Adj_hat[i*batch_size:-1]
            else:
                pred_reshape, pred, mean, logstd, lat_variable  = model(AdjacencyMatrix[i*batch_size:(i+1)*batch_size], FeaturesTensor_Normalize[i*batch_size:(i+1)*batch_size])
            
                labels = Adj_hat_reshape[i*batch_size:(i+1)*batch_size]   #Adj_hat[i*batch_size:(i+1)*batch_size] #
                current_Natoms = N_Atoms[i*batch_size:(i+1)*batch_size]

            if epoch == n_epochs:
                if i == 0:
                    final_lat_variable = lat_variable
                else:
                    final_lat_variable = torch.cat([final_lat_variable, lat_variable])
                

            
            # for nat in range(len(current_Natoms)):
            #     pred_Adj_hat_current =  pred[nat]
            #     #print(pred_Adj_hat_current.shape, int(current_Natoms[nat]))
            #     pred_Adj_hat_current_reduced = pred_Adj_hat_current[0:int(current_Natoms[nat]), 0:int(current_Natoms[nat])]

            #     original_Adj_hat_current         = labels[nat]
            #     original_Adj_hat_current_reduced = original_Adj_hat_current[0:int(current_Natoms[nat]), 0:int(current_Natoms[nat])]

            #     pred_Adj_hat_current_reduced_reshape = torch.reshape(pred_Adj_hat_current_reduced, (-1,))
            #     original_Adj_hat_current_reduced_reshape = torch.reshape(original_Adj_hat_current_reduced, (-1,))

            #     # print(pred_Adj_hat_current_reduced_reshape.shape, original_Adj_hat_current_reduced_reshape.shape)
            #     if nat == 0:
            #         loss = MyLoss(pred_Adj_hat_current_reduced_reshape, original_Adj_hat_current_reduced_reshape)    
            #     else:
            #         loss += MyLoss(pred_Adj_hat_current_reduced_reshape, original_Adj_hat_current_reduced_reshape)


                



        # # # -- loss
            loss = MyLoss(pred_reshape, labels)
        # loss = norm*MyLoss(model(adjs[i*batch_size:(i+1)*batch_size], sigs[i*batch_size:(i+1)*batch_size])[0],labels)

        # kl_divergence = 0.5/ pred.size(0) * (1 + 2*logstd - mean**2 - torch.exp(logstd)**2).sum(1).mean()
        # loss -= kl_divergence

        # -- optimize
            loss.backward()
            MyOptimizer.step()

            cum_loss += loss.item()

        loss_epoch.append(cum_loss/10)

        print("Epoch #",epoch)

        

    
    
    return n_epochs, loss_epoch, final_lat_variable


# In[8]:


n_epochs, loss_x_epoch, final_lat_variable = Training_VGAE(Adjs_FWs_Tensor, Features_Tensor, NAtoms_FWs, 100, 25, 1e-4)


# In[9]:


print(final_lat_variable.shape)


# In[10]:


# -- plot loss
X = np.arange(1, n_epochs+1)
fig,ax = plt.subplots(figsize=(10,5))
ax.plot(X, loss_x_epoch,'r--', lw=1.5)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('epoch', fontsize=15)
ax.set_ylabel('BCE Loss', fontsize=15)
plt.title("Loss Function of a set of Zeolite Frameworks", fontsize = 20)
plt.show()


# In[29]:


final_lat_variable_aux = final_lat_variable

ndim, mdim, pdim = final_lat_variable_aux.shape

ListColumns = ["NameFW"]

for j in range(mdim):
    Var_aux = "Var_{j}".format(j = j)
    ListColumns.append(Var_aux)

import pandas as pd 
Framework_df = pd.DataFrame(columns=ListColumns)


for i in range(ndim):
    current_array = Name_FWs[i]
    current_array = np.array(current_array)
    current_array_aux = final_lat_variable_aux[i]
    current_array_aux = current_array_aux[:,0].detach().numpy()

    current_array = np.append(current_array, current_array_aux)

    Framework_df.loc[len(Framework_df.index)] = current_array


# In[31]:


Framework_df.to_csv("Dataframe_FWs_toPCADimensionAnalysis_kMeans.csv")

