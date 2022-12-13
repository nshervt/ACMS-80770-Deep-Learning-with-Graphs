"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 4: Programming assignment
Problem 1
"""
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=UserWarning)


class kernel:
    def __init__(self, K, R, d, J, lamb_max):
        # -- filter properties
        self.R = float(R)
        self.J = J
        self.K = K
        self.d = d
        self.lamb_max = torch.tensor(lamb_max)

        # -- Half-Cosine kernel
        self.a = R*torch.log(self.lamb_max) / (J-R+1)
        self.g_hat = lambda lamb: sum([self.d[k]*torch.cos(2*torch.pi*k*(lamb/self.a+0.5))*(-lamb>=0 and -lamb<self.a) for k in range(K+1)])

    def wavelet(self, lamb, j):
        """
            constructs wavelets ($j\in [2, J]$).
        :param lamb: eigenvalue (analogue of frequency).
        :param j: filter index in the filter bank.
        :return: filter response to input eigenvalues.
        """
        assert(j>=2 and j<=self.J)
        return self.g_hat(torch.log(torch.tensor(lamb)) - self.a*(j-1)/self.R)

    def scaling(self, lamb):
        """
            constructs scaling function (j=1).
        :param lamb: eigenvalue (analogue of frequency).
        :return: filter response to input eigenvalues.
        """
        g2 = self.R/2*sum([di**2 for di in self.d])
        g3 = sum([self.wavelet(lamb,i)**2 for i in range(2,self.J+1)])
        gj = torch.sqrt(self.R*self.d[0]**2 + g2 - g3)
        return gj


# -- define filter-bank
lamb_max = 2
J = 8
filter_bank = kernel(K=1, R=3, d=[0.5, 0.5], J=J, lamb_max=lamb_max)

# -- plot filters

x  = np.arange(0,2,.01)
y1 = np.array([filter_bank.scaling(xi) for xi in x])
plt.plot(x,y1, label='j=1')

for j in range(2,9):
    y2 = np.array([filter_bank.wavelet(xi,j=j) for xi in x])
    plt.plot(x,y2, label='j='+str(j))
plt.ylabel('filters')
plt.xlabel('lambda')
plt.legend()
plt.savefig('./Homework4/filters.png')
