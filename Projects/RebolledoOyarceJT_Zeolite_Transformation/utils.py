import torch
import numpy as np


def Get_AdjNormalized(Adj_Matrix):

    n = Adj_Matrix.shape[-1]
    D_negative_sqrt = torch.diag_embed(1/(Adj_Matrix+torch.eye(n)).sum(dim=2).sqrt())
    AdjMatrix_norm = D_negative_sqrt@(Adj_Matrix+torch.eye(n))@D_negative_sqrt
    
    return AdjMatrix_norm


def Reconstruction_AdjsMatrix(input):
    n_graphs = input.shape[0]

    input_tranpose = torch.transpose(input,1,2)
    a_aux = input@input_tranpose

    a_aux = torch.reshape(a_aux,(n_graphs,-1))

    a_aux = a_aux.sigmoid()

    return a_aux


def Get_AdjHat(Adj_Matrix):

    n = Adj_Matrix.shape[-1]
    Adj_hat = Adj_Matrix+torch.eye(n)
    
    return Adj_hat
    