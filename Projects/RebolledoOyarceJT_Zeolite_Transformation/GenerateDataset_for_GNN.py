import numpy as np
import networkx as nx
from Get_FW_Nodes_Edges import *
from ase.db import connect

from GetFeatures_FW import *
import torch
from torch import nn
from torch import optim

import matplotlib.pyplot as plt

from ase.db import connect
from GetFeatures_FW import *
from VGAE_Model import *
from utils import *

AllFWs_Name = []


df_Frameworks = connect('Database_FW.db')

it = 0

for row in df_Frameworks.select():
    Adjs_Matrix_aux = np.zeros((1440*1440))
    SOAP_Matrix_aux = np.zeros((1440*252))

    atoms = row.toatoms()
    nameFW = row.NameFramework



    NumberElements = np.array(atoms.get_chemical_symbols())
    Cartesian_coordinates_all = np.array(atoms.get_positions())
    # NumberElements = np.array(atoms.get_chemical_symbols())

    # Identify the index of Oxygen and Silicon
    Index_Element_Si  = np.where(NumberElements == 'Si')[0].copy()
    Index_Element_O   = np.where(NumberElements == 'O')[0].copy()
    Index_Element_Al  = np.where(NumberElements == 'Al')[0].copy()



    Index_Element_Si_Al = Index_Element_Si.copy()
    if len(Index_Element_Al) > 0: 
        for i in Index_Element_Al:
            Index_Element_Si_Al = np.append(Index_Element_Si_Al, i)          



    Edges_total = Get_Edges(atoms)

    labelNodes = []
    Cartesian_coordinates_Si_Al = []
    for i,T in enumerate(Index_Element_Si_Al):
        Cartesian_coordinates_Si_Al.append(list(Cartesian_coordinates_all[T]))
        labelNodes.append("At." + str(i))

    G = nx.Graph()
    G.add_nodes_from(Index_Element_Si_Al)
    G.add_edges_from(Edges_total)

    A = nx.to_numpy_matrix(G) #nx.adjacency_matrix(G)


    atoms_copy = atoms.copy()
    del atoms_copy[[atom.index for atom in atoms_copy if atom.symbol=='O']] 
    N = len(atoms_copy)


    soap_values = get_SOAPDescriptors(atoms_copy)

    adjs_n, adjs_m = A.shape
    SOAP_n, SOAP_m = soap_values.shape

    A_reshape = A.reshape(-1)
    soap_values_reshape = soap_values.reshape(-1)

    # Adjs_Matrix_aux[0:adjs_n,0:adjs_m] = A
    # SOAP_Matrix_aux[0:SOAP_n,0:SOAP_m] = soap_values

    Adjs_Matrix_aux[0:(adjs_n*adjs_m)] = A_reshape[0,:]
    SOAP_Matrix_aux[0:(SOAP_n*SOAP_m)] = soap_values_reshape[:]

    if len(AllFWs_Name) == 0:
        AllFWs_Name.append(nameFW)
        Adjs_FWs = Adjs_Matrix_aux
        Tensor_FWs = SOAP_Matrix_aux
    elif len(AllFWs_Name) == 1:
        AllFWs_Name.append(AllFWs_Name)
        Adjs_FWs = np.array([Adjs_FWs, Adjs_Matrix_aux])
        Tensor_FWs = np.array([Tensor_FWs, SOAP_Matrix_aux])
    else:
        AllFWs_Name.append(nameFW)
        Adjs_FWs = np.append(Adjs_FWs, [Adjs_Matrix_aux], axis = 0)
        Tensor_FWs = np.append(Tensor_FWs, [SOAP_Matrix_aux], axis = 0)
    
    print(nameFW, it)

    it += 1



np.savetxt('data_features/data_TensorFWs.csv', Tensor_FWs, delimiter=',')
np.savetxt('data_features/data_AdjsFWs.csv', Adjs_FWs, delimiter=',')

print("Ok Reading CSV Files")