# accuracy evaluations

import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from numpy import genfromtxt as gxt
import matplotlib
import matplotlib.ticker as mtick

# plotting specs
fs = 24
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)

# histogram plotting
# inputs:
#        Pth: where to save the fig
#        X: inputs
#        xlabel: xlabel
#        c: color
#        exact: expression of the exact distribution as a lambda function
def Hist_plot(Pth, X, xlabel, c = 'b', exact=None):

	os.makedirs(Pth,exist_ok = True)

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111)
	n, bins, patches = plt.hist(X, color=c, alpha = 0.5)
	
	if exact != None:
		plt.plot(bins, exact(bins), 'k', linewidth=2, alpha=0.5)

	plt.grid('on')
	plt.xlabel(xlabel,fontsize=fs)
	plt.ylabel('Counts',fontsize=fs)
	plt.tick_params(labelsize=fs)
	ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))

	fig_name = Pth + '/'+ xlabel +'.png'
	plt.savefig(fig_name)
	return 0



prefix = 'Model_save/Quarter2D/' # case-specific prefix
num_part    = 256  # number of partitions of the high order mesh (high fidelity mesh)

acc_before = []
acc_after  = []
loss_training = []
loss_testing  = []

for i in range(num_part):
	model_name = prefix + 'Rank=' + str(i) + '-GNN-nB-64-Lr-0.005/'
	acc_before.append(gxt(model_name + 'accuracy_before.csv', delimiter=','))
	acc_after.append(gxt(model_name + 'accuracy_after.csv', delimiter=','))
	loss_training.append(gxt(model_name + 'MSE Loss-train.csv', delimiter=','))
	loss_testing.append(gxt(model_name + 'MSE Loss-test.csv', delimiter=','))


#-------------------------------hist----------------------------------------#
acc_before = np.array(acc_before)
acc_after = np.array(acc_after)
Hist_plot(prefix, acc_before, r'$\zeta_1$', c = 'b')
Hist_plot(prefix, acc_after, r'$\zeta_2$', c = 'r')
#---------------------------------------------------------------------------#


# #-----------------------------training acc----------------------------------#
# loss_training = np.array(loss_training).mean(axis=0)
# loss_testing = np.array(loss_testing).mean(axis=0)

# fig = plt.figure(figsize=(10,8))
# plt.semilogy(loss_training, '-b', linewidth=2, label = 'Training')
# plt.semilogy(loss_testing, '-r', linewidth=2, label = 'Testing')

# matplotlib.rc('font', size=fs+2)
# plt.xlabel(r'$\textrm{Epoch}$',fontsize=fs)
# plt.ylabel(r'$\bar{\mathcal{L}}$',fontsize=fs)
# plt.tick_params(labelsize=fs+2)
# plt.legend(fontsize=fs-3)
# plt.show()
# #---------------------------------------------------------------------------#