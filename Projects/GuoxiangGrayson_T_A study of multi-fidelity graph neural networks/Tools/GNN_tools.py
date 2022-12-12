import torch
import torch.nn as nn
import numpy as np
import meshio
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


# mesh information extraction for 2D meshes
# Inputs:
#        mesh_name: path of the math file
# Outputs:
#        A      : Adjacency matrix
#        A_dic     : neighbor information of each node, saved as a dictionary

def Adj2D(mesh_name):
	
	Mesh      = meshio.read(mesh_name)	
	Cells     = (Mesh.cells_dict)['triangle']   # tri3 cells
	Points    =  Mesh.points                    # nodal points
	nELE      =  len(Cells[:,0])    			# number of element

	# construct adj matrix, assuming un-directed graph
	V = len(Points)     # number of nodes
	A = np.zeros((V,V), dtype=int) # initialize adj matrix
	
	# loop tho elements
	for i in range(nELE):
		# locate an element
		ele = Cells[i]
		# loop vertices of the current element
		for p in ele:
			neighbors = np.setdiff1d(ele,p)
			# loop neighbors of the current vertex
			for q in neighbors:
				A[p,q] = 1
				A[q,p] = 1 # we are considering undirected graph

	# loop thorough nodes
	A_dic = {}
	for i in range(len(Points)):
		# find neighbors of node i
		neighbors = np.where(A[i,:] == 1)[0]
		A_dic[str(i)] = neighbors

	return A, A_dic



#-----------------------------------------------------------#
# training-testing dataset random split
# Input: 
#       samplesize: how many samples in total
#       T_portion: how much for gradient update
#       X: model input tensor
#       Y: model output tensor (to be learned)
# Output:
#       sliced tensors
def TT_split(Sample_size, T_portion, X, Y):
    
    train_length = range(Sample_size)

    # pick random, un-ordered number from the range of all samples
    train_slice  = np.random.choice(train_length, size=int(T_portion*Sample_size), replace=False) 
    # do a set difference to get the test slice, this is random but ordered
    test_slice   = np.setdiff1d(train_length, train_slice) 
    
    #-------making slices out of random choices-------#
    # Note: we always use the 0-th dimension as the batch dimension
    train_tensor        = X[train_slice,:]
    train_truth_tensor  = Y[train_slice,:]

    test_tensor        = X[test_slice,:]
    test_truth_tensor  = Y[test_slice,:]

    return train_tensor, train_truth_tensor, test_tensor, test_truth_tensor
#-----------------------------------------------------------#


# auto-batching tools for regression 
class MyDatasetXY(Dataset):
    def __init__(self, X, Y):
        super(MyDatasetXY, self).__init__()
        
        # sample size checker, the first dimension is always the batch size
        assert X.shape[0] == Y.shape[0]
        
        self.X = X
        self.Y = Y

    # number of samples to be batched
    def __len__(self):
        return self.X.shape[0] 
       
    # get samples, third arg gives use the label
    def __getitem__(self, index):
        return self.X[index], self.Y[index], index



# Nonlinear MLP function, can be generalized for various purposes
# inputs:         
        # NI: input size
        # NO: ouput size
        # NN: hidden size
        # NL: num of hidden layers
        # act: type of nonlinear activations, default: relu
# output:
#       sequential of layers

def MLP_nonlinear(NI,NO,NN,NL,act='relu', BN=False):

    # select act functions
    if act == "relu":
        actF = nn.ReLU()
    elif act == "tanh":
        actF = nn.Tanh()
    elif act == "sigmoid":
        actF = nn.Sigmoid()
    elif act == 'leaky':
        actF = nn.LeakyReLU()

    #----------------construct layers----------------#
    MLP_layer = []

    # Input layer
    MLP_layer.append( nn.Linear(NI, NN) )
    MLP_layer.append(actF)
    
    # Hidden layer, if NL < 2 then no hidden layers
    for ly in range(NL-2):
        MLP_layer.append(nn.Linear(NN, NN))
        MLP_layer.append(actF)
        
        if BN == True:
            #-----------------------------------------------------#
            MLP_layer.append(nn.BatchNorm1d(NN))
            #-----------------------------------------------------#

    # Output layer
    MLP_layer.append(nn.Linear(NN, NO))
    
    # seq
    return nn.Sequential(*MLP_layer)


# max-min scaler in the training stage
def max_min_scaling(X):
    scale_max = X.max() # locate max 
    scale_min = X.min() # locate min

    # scale to [-1,1]
    X_scaled = 2.*(X-scale_min)/(scale_max-scale_min) - 1
    return scale_min, scale_max, X_scaled

# use standardized normalization method to scale the dataset
# input:
#      X: dataset to be scaled
#      device: targeted device
# Output:
#      mu: mean per example
#      std: standard deviation per example
#      X_scaled = (X-mu)/std

# Note: example means one partition
def standard_scaling(X, device):
    
    # init
    mu  = torch.zeros(X.shape[1]).to(device)
    std = torch.zeros(X.shape[1]).to(device)

    # taking mean and std at the example direction, one mu and one sigma per feature
    for i in range(X.shape[1]):
        mu[i]  = torch.mean(X[:,i])
        std[i] = torch.std(X[:,i])

    # normalize each feature by its own mean and std
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - mu[i])/std[i]
    return mu, std, X

# scale the solution back to original scale,
def scale_it_back(X, mu, std):

    for i in range(X.shape[1]):
        X[:,i] = X[:,i] * std[i] + mu[i]

    return X



# general function to plot training and testing curves
# Inputs:
#        path: where to save the data
#        training: traing losses
#        testing:  testing losses
#        args: plotting args

def TT_plot(PATH, training, testing, ylabel, yscale = 'normal' ):

    # plotting specs
    fs = 24
    plt.rc('font',  family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text',  usetex=True)

    # plot loss curves
    fig1 = plt.figure(figsize=(10,8))
    
    if yscale == 'semilogy':
        plt.semilogy(training, '-b', linewidth=2, label = 'Training')
        plt.semilogy(testing, '-r', linewidth=2, label = 'Testing')
    else:
        plt.plot(training, '-b', linewidth=2, label = 'Training')
        plt.plot(testing, '-r', linewidth=2, label = 'Testing')
    
    plt.xlabel(r'$\textrm{Epoch}$',fontsize=fs)
    plt.ylabel(ylabel,fontsize=fs)
    plt.tick_params(labelsize=fs+2)
    plt.legend(fontsize=fs-3)
       
    # save the fig   
    fig_name = PATH + '/'+ ylabel +'.png'
    plt.savefig(fig_name)


    # save the data
    train_name   = PATH + '/' + ylabel + '-train.csv'
    test_name    = PATH + '/' + ylabel + '-test.csv'
    
    np.savetxt(train_name, training,   delimiter = ',')
    np.savetxt(test_name, testing,   delimiter = ',')
            
    return 0