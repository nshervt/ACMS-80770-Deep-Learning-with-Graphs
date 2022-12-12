# multi-fidelity GNN
#	Note: the low-fid solution is projected or interpolated to the high-fid solution
#		  the training on the partitioned high-fid mesh, each piece is obtained via parallel mesh partitioning	
# method: classical Message-passing GNN
# training method: mini-batch gradient descent via Adam

import torch
import torch.nn as nn
import os
from mpi4py import MPI
import math
from Tools.GNN_tools import *
import numpy as np
from Tools.Model import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # determine if to use gpu
device = 'cpu'
#----------------------------------DATA PREPARE-----------------------------------------#
prefix = 'Canti2D/' # case-specific prefix
num_part    = 128   # number of partitions of the high order mesh (high fidelity mesh)

# sub-connectivity matrix build-up
mesh_name = prefix + 'Partitioned_Mesh/Rank-'+str(rank) + '.vtk' # mesh part save
Adj, Adj_dict       = Adj2D(mesh_name)   # adjacency matrix (and the associated dictionary) out of the current mesh part
# tensorize
Adj = torch.from_numpy(Adj).to(device)
local_nodes = np.genfromtxt(prefix + 'Partitioned_Mesh/Rankwised_Nodes/Rank='+ \
											str(rank)+'_local_nodes.csv', delimiter=',').astype(int)
	
#-----------------------------------------#
if rank == 0:
	print( 'Local Connectivity done.....')
#-----------------------------------------#

# loading computed dataset (global solution)
X_name = prefix + 'Sol/low-to-high000000.vtu' # interpolated solution from fenics
X_dic  = meshio.read(X_name)                  # dictionary of interpolated dataset
X_emb  = (X_dic.point_data)['f_27'][:,:2]     # displacement solution as node embeddings
X_geo  = (X_dic.points)[:,:2]                 # geometric information (coordinates)

Y_name = prefix + 'Sol/high-fid000000.vtu'    # high-fid solution from fenics
Y_dic  = meshio.read(Y_name)                  # dictionary of high-fid solution
Y_emb  = (Y_dic.point_data)['f_8'][:,:2]      # displacement solution as node embeddings

# construct embeddings
# 4: means ux, uy, x, y of the low-fid solution
# 2: means ux, uy of the high-fid solution
Node_embedding       = np.zeros((len(Adj), 4))
Node_embedding_truth = np.zeros((len(Adj), 2))

# locate features from the global dataset
for i in range(len(Adj)):

	global_id = local_nodes[i] # locate global id
	
	# insert node embeddings
	Node_embedding[i,:2]       = X_emb[global_id,:]
	Node_embedding[i,2:]       = X_geo[global_id,:]
	Node_embedding_truth[i,:]  = Y_emb[global_id,:]

# rename
X = torch.from_numpy(Node_embedding).float().to(device)  
Y = torch.from_numpy(Node_embedding_truth).float().to(device)

#-----------------------------------------#
if rank == 0:
	print( 'Node embedding constructred.....')
#-----------------------------------------#

# accuracy before implement the model
Acc_before = 1 - torch.norm(X[:,:2]-Y,2)/torch.norm(Y,2)

# max-min scalar
# scale_minX, scale_maxX, X = max_min_scaling(X)
# scale_minY, scale_maxY, Y = max_min_scaling(Y)

# z-standardization scalar
mu_X, std_X, X = standard_scaling(X, device)
mu_Y, std_Y, Y = standard_scaling(Y, device)


# dataset split
# Note: based our training algorithm: loop over all nodes and gather neighborhood information
# 		to do message passing. Thus, we take random sample of nodes 

T_portion   = 0.75    		      # part of the data used for training
train_tensor, train_truth_tensor, test_tensor, test_truth_tensor = \
											TT_split(len(Adj), T_portion, X, Y)

# print('rank = '+ str(rank) + ', ' + str(train_tensor.shape) + ', ' + str(train_truth_tensor.shape))
# print('rank = '+ str(rank) + ', ' + str(test_tensor.shape) + ', ' + str(test_truth_tensor.shape))

#------------------------------------------------------------------------------------------------------#


# start to build networks
msg   = 1                                # number of msg rounds
EI, EO, NI, NO  = 4,4,2,4                # edge input, edge output, node input, node output
for batch_size in [64]:                  # search optimal batch size
	for learning_rate in [5e-3]:         # search optimal learning rate

		# create folder to save the trained model
		Ori_PATH = 'Model_save/' 
		
		# save name
		Save_name = prefix + 'Rank='+str(rank)+'-GNN-nB-'+ str(batch_size) + '-Lr-' + str(learning_rate)
		
		PATH = Ori_PATH + Save_name
		os.makedirs(PATH,exist_ok = True)

		# other hyper-para spec
		lr_min      = 5e-6                  # keep the minimal learning rate the same, avoiding updates that is too small
		decay       = 0.999                 # learning rate decay rate
		
		# call model
		model = Message_passing_GNN_parallel(device, msg, EI, EO, NI, NO )
		
		# convert to GPU
		model = model.to(device) 
	
		# standard loss function
		criterion = nn.MSELoss()

		# define optimizer
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

		# build lr scheduler
		lambda1 = lambda epoch: decay ** epoch # lr scheduler by lambda function
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) # define lr scheduler with the optim

		# calculate corresponding epoch number
		num_epochs = int(math.log(lr_min/learning_rate, decay))

		#------------------------------------------------------------------------------------------------------#
		if rank == 0:
			print('Number of epoch  is: ' + str(num_epochs))
		#------------------------------------------------------------------------------------------------------#

		# loss and accuracy saves placeholders
		#----------------------------------------------------#
		train_save   = [] # epoch-wise training loss save 
		test_save    = [] # epoch-wise testing loss save 

		train_acc_save   = [] # epoch-wise training accuracy save 			
		test_acc_save    = [] # epoch-wise testing accuracy save
		#----------------------------------------------------# 

		#-------------Start-training-validation-----------#
		for epoch in range(num_epochs):

			#----------------------use dataloader to do auto-batching--------------------------------#
			traindata    =  MyDatasetXY(train_tensor, train_truth_tensor)
			trainloader  =  torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
	
			testdata     =  MyDatasetXY(test_tensor, test_truth_tensor)
			testloader   =  torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False)
			
			num_batches_train = len(trainloader) # total number of training mini batches
			num_batches_test  = len(testloader)  # total number of testing mini batches
			#----------------------------------------------------------------------------------------#

			if epoch == 0 and rank == 0:
				print('total number of training and testing batches are: '+\
										 	str(num_batches_train) + ',' +  str(num_batches_test))


			#--------------------------------------Training step------------------------------------------# 
			X_temp = X
			model, train_loss_epoch, train_acc_epoch = \
				Training( device, model, trainloader, criterion, optimizer, Adj, X_temp )
										
			# save training results
			train_save.append(train_loss_epoch/num_batches_train)
			train_acc_save.append(train_acc_epoch/num_batches_train)

			# print out values for training stats
			if epoch%50 == 0 and rank == 0:
				print("Training: Epoch: %d, mse loss: %1.5e" % (epoch, train_save[epoch]) ,\
					", mse acc : %1.5e" % (train_acc_save[epoch]), \
					', lr=' + str(optimizer.param_groups[0]['lr']))

			# update learning rate
			scheduler.step()

			#----------------------------------Testing step--------------------------------------------#
			test_loss_epoch, test_acc_epoch =  \
					Testing(device, model, testloader, criterion, Adj, X_temp)

			# save quantities
			test_save.append(test_loss_epoch/num_batches_test)
			test_acc_save.append(test_acc_epoch/num_batches_test)

			# print out values for validation stats
			if epoch%50 == 0 and rank == 0:
				print("Testing: Epoch: %d, mse loss: %1.5e" % (epoch, test_save[epoch]) ,\
					", mse acc : %1.5e" % (test_acc_save[epoch]))


		#---------------------------save the model-------------------------------#
		model_save_name   = PATH + '/model.pth'
		torch.save(model.state_dict(), model_save_name)

		train_save = (torch.FloatTensor(train_save)).cpu()
		test_save  = (torch.FloatTensor(test_save)).cpu()

		train_acc_save = (torch.FloatTensor(train_acc_save)).cpu()
		test_acc_save  = (torch.FloatTensor(test_acc_save)).cpu()

		TT_plot(PATH, train_save, test_save, 'MSE Loss', yscale = 'semilogy' )
		TT_plot(PATH, train_acc_save, test_acc_save, 'MSE Accuracy')

		# model evaluation
		Acc_after = Acc_evaluation(device, model, Adj, X_temp, Y, criterion, mu_Y, std_Y)
		# save the computed accuracy
		np.savetxt(PATH+'/accuracy_before.csv', [Acc_before], delimiter=',')
		np.savetxt(PATH+'/accuracy_after.csv', [Acc_after.item()], delimiter=',')



