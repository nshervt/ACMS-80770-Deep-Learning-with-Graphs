"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 4: Programming assignment
Problem 2
"""
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=UserWarning)

from torch import nn
from rdkit import Chem
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor


# -- load data
class MolecularDataset(Dataset):
    def __init__(self, N, train=True):
        if train:
            start, end = 0, 100000
        else:
            start, end = 100000, 130000


        dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True),
                                                   return_smiles=True,
                                                   target_index=np.random.choice(range(133000)[start:end], N, False))

        self.atom_types = [6, 8, 7, 9, 1]
        self.V = 9

        self.adjs = torch.stack(list(map(self.adj, dataset)))
        self.sigs = torch.stack(list(map(self.sig, dataset)))
        self.prop = torch.stack(list(map(self.target, dataset)))[:, 5]
        self.prop_2 = torch.stack(list(map(self.target_2, dataset_smiles)))

    def target_2(self, smiles):
        """
            compute the number of hydrogen-bond acceptor atoms
        :param smiles: smiles molecular representation
        :return:
        """
        mol = Chem.MolFromSmiles(smiles)

        return torch.tensor(Chem.rdMolDescriptors.CalcNumHBA(mol))

    def adj(self, x):
        x = x[1]
        adjacency = np.zeros((self.V, self.V)).astype(float)
        adjacency[:len(x[0]), :len(x[0])] = x[0] + 2 * x[1] + 3 * x[2]
        return torch.tensor(adjacency)

    def sig(self, x):
        x = x[0]
        atoms = np.ones((self.V)).astype(float)
        atoms[:len(x)] = x
        out = np.array([int(atom == atom_type) for atom_type in self.atom_types for atom in atoms]).astype(float)
        return torch.tensor(out).reshape(5, len(atoms)).T

    def target(self, x):
        """
            return Highest Occupied Molecular Orbital (HOMO) energy
        :param x:
        :return:
        """
        x = x[2]
        return torch.tensor(x)

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, item):
        return self.adjs[item], self.sigs[item], self.prop[item], self.prop_2[item]


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
        lamb_min = np.exp(self.a*((j-1)/self.R-1))
        if  lamb < lamb_min:
            # print(lamb)
            lamb = lamb_min
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


class scattering(nn.Module):
    def __init__(self, J, L, V, d_f, K, d, R, lamb_max):
        super(scattering, self).__init__()

        # -- graph parameters
        self.n_node = V
        self.n_atom_features = d_f

        # -- filter parameters
        self.K = K
        self.d = d
        self.J = J
        self.R = R
        self.lamb_max = lamb_max
        self.filters = kernel(K=1, R=3, d=[0.5, 0.5], J=8, lamb_max=2)

        # -- scattering parameters
        self.L = L

    def compute_spectrum(self, W):
        """
            Computes eigenvalues of normalized graph Laplacian.
        :param W: tensor of graph adjacency matrices.
        :return: eigenvalues of normalized graph Laplacian
        """

        # -- computing Laplacian
        # W = W + torch.eye(len(W))
        D = torch.diag_embed(W.sum(1))
        L = D - W

        # -- normalize Laplacian
        diag  = W.sum(1)
        dhalf = torch.diag_embed(1. / torch.sqrt(torch.max(torch.ones(diag.size()), diag)))
        # print(torch.isnan(dhalf))
        L     = dhalf @ L @ dhalf

        # -- eig decomposition
        E, V = torch.symeig(L, eigenvectors=True)
        
        return abs(E), V

    def filtering_matrices(self, W):
        """
            Compute filtering matrices (frames) for spectral filters
        :return: a collection of filtering matrices of each wavelet kernel and the scaling function in the filter-bank.
        """

        filter_matrices = []
        E, V = self.compute_spectrum(W)
        # print(E)

        # -- scaling frame
        LM = torch.diag_embed(torch.tensor([self.filters.scaling(e) for e in E]))
        filter_matrices.append(V @ LM @ V.T)

        # -- wavelet frame
        for j in range(2, self.J+1):
            LM = torch.diag_embed(torch.tensor([self.filters.wavelet(e, j=j) for e in E])).type(torch.float64)
            filter_matrices.append(V @ LM @ V.T)
            # filter_matrices.append(V @ ... @ V.T)

        return torch.stack(filter_matrices)

    def forward(self, W, f):
        """
            Perform wavelet scattering transform
        :param W: tensor of graph adjacency matrices.
        :param f: tensor of graph signal vectors.
        :return: wavelet scattering coefficients
        """

        # -- filtering matrices
        g = self.filtering_matrices(W)

        # --
        U_ = [f]

        # -- zero-th layer
        # S = ...  # S_(0,1)
        S = f.mean(0)

        for l in range(self.L):
            U = U_.copy()
            U_ = []

            for f_ in U:
                for g_j in g:

                    U_.append(abs(g_j@f_))
                    S_li = torch.mean(abs(g_j@f_),0)

                    # -- append scattering feature S_(l,i)
                    S = torch.cat((S,S_li), dim=0)

        return S


# -- initialize scattering function
scat = scattering(L=2, V=9, d_f=5, K=1, R=3, d=[0.5, 0.5], J=8, lamb_max=2)


# -- load data
training_points = 5000
testing_points  = 1000
data = MolecularDataset(N=training_points + testing_points)

# -- Compute scattering feature maps

S = torch.cat([scat.forward(data.adjs[b], data.sigs[b]).unsqueeze(0) for b in range(len(data.adjs))])

# -- PCA projection
pca    = PCA(n_components=2)
latent = pca.fit_transform(S)

# -- plot feature space
plt.close('all')
plt.scatter(latent[:,0], latent[:,1], c=data.prop_2)
plt.colorbar()
plt.savefig('./Homework4/latent_plot.png')
# torch.save(S,'S_tensor.pt')

# -- Neural network to predict HOMO

class FCNN(nn.Module):
    """
        Neural network to predict HOMO
    """
    def __init__(self, in_features = 365, hidd_features = 256):
        super(FCNN, self).__init__()
        self.nn = nn.Sequential(nn.Linear(in_features,hidd_features),
                                nn.LeakyReLU(),
                                nn.Linear(hidd_features,1))

    def forward(self, S):
        return self.nn(S)

# S = torch.load('S_tensor.pt')
S = S.type(torch.float)

targ = data.prop[:training_points][:,None]

mean_in  = S.mean(0)
std_in   = S.std(0)+1e-6
mean_out = targ.mean()
std_out  = targ.std()+1e-6

model = FCNN()
MyLoss = nn.MSELoss()
MyOptimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(MyOptimizer,factor=0.9, patience=100, min_lr=1e-4)
error = []

# -- update parameters
for epoch in range(5000):
        
    # -- predict
    # pred = model(S[:training_points])
    pred = model((S[:training_points]-mean_in)/std_in)*std_out + mean_out

    # -- loss
    loss = MyLoss(pred, targ)
    loss.backward()
    
    # -- optimize
    MyOptimizer.step()
    MyOptimizer.zero_grad()
    # scheduler.step(loss.item())
    
    error.append(loss.item())

    if epoch%100 == 0:
        print("epoch ",epoch, loss.item())
        plt.close('all')
        plt.plot(pred.detach().numpy(), targ,  '^')
        x = torch.linspace(torch.min(targ),torch.max(targ),100)
        plt.plot(x,x)
        plt.ylabel('Traget')
        plt.xlabel('Prediciton')
        plt.xlim(min(targ.detach().numpy()),max(targ.detach().numpy()))
        plt.savefig('./Homework4/train.png')

    
# -- plot loss
error = np.array(error)

"""
Test
"""
pred = model((S[training_points:]-mean_in)/std_in) *std_out + mean_out
pred = pred.detach().numpy() 
targ = data.prop[training_points:]

plt.close('all')
plt.plot(pred, targ, '^')
x = torch.linspace(torch.min(targ),torch.max(targ),100)
plt.plot(x,x)
plt.xlim(min(pred),max(pred[:,0]))
plt.ylabel('Traget')
plt.xlabel('Prediciton')
plt.savefig('./Homework4/test.png')


plt.close('all')
plt.plot(error)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.savefig('./Homework4/loss.png')
