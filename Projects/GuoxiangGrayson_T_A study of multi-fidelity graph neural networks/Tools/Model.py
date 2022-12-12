import torch
import torch.nn as nn
import numpy as np
from Tools.GNN_tools import *


# The main message passing class
class Message_passing_GNN_parallel(nn.Module):
	
	def __init__(self, device, msg, EI, EO, NI, NO ):
		
		super().__init__()               # allow inherience from nn.Module

		self.device = device 
		self.msg    = msg                # number of msg passing rounds

		self.edge_input      = EI        # edge input
		self.edge_output     = EO        # edge output
		self.node_input      = NI        # node input
		self.node_output     = NO        # node output


		# define node function as fully connected mlps
		node_nn = 64
		node_nl = 4
		self.node_function_neighbor  = \
				MLP_nonlinear(self.node_input, self.node_output, node_nn, node_nl, act='relu', BN=False)
	
		# define edge function as fully connected mlps
		edge_nn = 64
		edge_nl = 4
		self.edge_function \
				 = MLP_nonlinear(self.edge_input, self.edge_output, edge_nn, edge_nl, act='relu', BN=False)

		# final nodal feature prediction
		final_nn = 64
		final_nl = 4
		self.node_function_self     =  \
				MLP_nonlinear(self.node_input + self.node_output + self.edge_output, \
									2, node_nn, node_nl, act='relu', BN=False)
#----------------------------------------------------------------------------------------------------------------#
	# message passing GNN forward function
	# Input:   X: batched features: ux, uy, x, y
	# 		   idx: example label, locate to the current mesh part
	#          Adj : adjacency matrix
	# Output:
	#        Y_hat : predicted node embedding of the high fid simulation

	def forward(self, X, idx, Adj, X_local):

		# prepare batched node feature
		batch_neighbor_idx = Adj[idx].float() # neighbor nodes of the current batch
		batch_num_neighbor = batch_neighbor_idx.sum(dim=1).reshape((-1,1))

		node_feature_agged = batch_neighbor_idx @ X_local[:,:2]  # aggregate neighbor information
		

		node_output    = self.node_function_neighbor(node_feature_agged) # neighor forward map

		# prepare batched edge feature
		batch_neighbor_idx_edge        = -1 * batch_neighbor_idx # negation, to prepare for the difference
		edge_agged                     = batch_neighbor_idx_edge @ X_local # aggregate edge information
		edge_feature_agged             = batch_num_neighbor * X  + edge_agged # get edge feature, the differences
		
		edge_output = self.edge_function(edge_feature_agged)

		self_feature = torch.cat((X[:,:2],node_output, edge_output), dim=1)

		self_output  = self.node_function_self(self_feature) # self forward map, updating the current node feature


		return self_output
#----------------------------------------------------------------------------------------------------------------#



# The training algorithm
# Inputs:
#       device: cpu or gpu
#       model : GNN model
#       trainloader: training dataset
#       criterion: loss function
#       optimizer: gradient algorithm
#       adj: local adjacency matrix
#       X_local  : local feature matrix

def Training( device, model, trainloader, criterion, optimizer, adj, X_local):

	# init loss
	train_loss, train_acc = 0,0

	# train status
	model.train()
	
	# batch loop
	for X,Y, idx in trainloader:

		# Zero-out the gradient
		optimizer.zero_grad()

		# forward the model to get node embedding prediction and edge embedding prediction
		Y_hat = model(X, idx, adj, X_local)

		# node embedding loss
		loss_mse = criterion(Y_hat, Y)
		
		# node embedding mse accuracy
		Rel_acc  = 1.0 - loss_mse/criterion(Y, torch.zeros_like(Y, device=device))

		# record the loss and acc
		train_loss += loss_mse.item()
		train_acc  += Rel_acc.item()

		# back-propagation
		loss_mse.backward()

		# gradient update
		optimizer.step()

	return model, train_loss, train_acc


# The testing algorithm
# Inputs:
#       device: cpu or gpu
#       model : GNN model
#       testloader: testing dataset
#       criterion: loss function
#       adj: local adjacency matrix
#       X_local  : local feature matrix
# Outputs:
#       test_loss: testing error of the whole batch
#       test_acc: testing accuracy of the whole batch

def Testing( device, model, testloader, criterion, adj, X_local):
	
	# init loss
	test_loss, test_acc = 0,0

	# train status
	model.eval()

	# batch loop
	with torch.no_grad():

		# batch loop
		for X,Y, idx in testloader:

			# forward the model to get node embedding prediction and edge embedding prediction
			Y_hat = model(X, idx, adj, X_local)

			# node embedding loss
			loss_mse = criterion(Y_hat, Y)
			
			# node embedding mse accuracy
			Rel_acc  = 1.0 - loss_mse/criterion(Y, torch.zeros_like(Y, device=device))

			# record the loss and acc
			test_loss += loss_mse.item()
			test_acc  += Rel_acc.item()

	return test_loss, test_acc


# model evaluation, currently only examine the accuracy
# Inputs:
#       device: cpu or gpu
#       model : trained GNN model
#       Adj: local adjacency matrix
#       X: input dataset
#       criterion: loss function
#       mu_Y: scaling constants: mean
#       std_Y: scaling constants: standard deviation
# Outputs:
#       relative MSE accuracy, in the original scale

def Acc_evaluation(device, model, Adj, X, Y, criterion, mu_Y, std_Y ):

	# train status
	model.eval()

	# batch loop
	with torch.no_grad():

		# forward the model as a whole batch
		Y_hat = model(X, torch.arange(len(X)), Adj, X)

		# scale it back
		Y_hat = scale_it_back(Y_hat, mu_Y, std_Y) # scale back prediction
		Y = scale_it_back(Y, mu_Y, std_Y)  # scale back truth

		# node embedding loss
		loss_mse = criterion(Y_hat, Y)	

		# node embedding mse accuracy
		Rel_acc  = 1.0 - loss_mse/criterion(Y, torch.zeros_like(Y, device=device))

	return Rel_acc
