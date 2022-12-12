from collections import defaultdict
import numpy as np
import sys
import timeit
import argparse

from rdkit import Chem
from rdkit.Chem import Draw

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

'''
Since we are using r-radius subgraph embedding, a function that can properly identify
r-radius subgraphs will process the raw data and feed it to post GNN training.
'''


def getsub(r, atoms, neighbor_bond_dict, subgraph_dict, edge_dict):
    """
    Extract the subgraphs from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """
    if (len(atoms) == 1) or (r == 0):
        # for cases that system has single nodes or closing the subgraph searching
        nodes = [subgraph_dict[a] for a in atoms]
    else:
        nodes = atoms
        sub_edge = neighbor_bond_dict

        for _ in range(r):
            # updating subgraphs' id based on atom ids and edges with in r
            node_list = []
            for i, j in sub_edge.items():
                neighbors = [(nodes[k], e) for k, e in j]  # identify the neighbors based on the neighbor_bond_dict
                subgraph = (nodes[i], tuple(sorted(neighbors)))  # extract subgraphs from neighbor list
                node_list.append(subgraph_dict[subgraph])

            # updating edges' id
            sub_edge_dict = defaultdict(lambda: [])
            for i, j in sub_edge.items():
                for k, e in j:
                    side = tuple(sorted((nodes[i], nodes[k])))  # find a pair of atoms
                    e = edge_dict[(side, e)]  # search the edges between the pair
                    sub_edge_dict[i].append((k, e))  # update the edge_dict

            nodes = node_list  # node_list -> nodes (subgraphs)
            sub_edge = sub_edge_dict
    return np.array(nodes)


def add_pbc(atoms, radius, neighbor_bond_dict, subgraph):
    """
    Use local searching and expansion to manipulate subgraph vectors
    """
    b = [x for x, y in list(enumerate(atoms)) if y == 0]
    if len(b) != 2:
        return subgraph
    else:
        left = b[0]
        right = b[1]
        # dynamical searching for expanding fragments
        unit = [atoms[i] for i in range(len(atoms)) if (i != left and i != right)]
        neighbor_list = []
        for i in range(len(atoms)):
            neighbor_list.append([neighbor_bond_dict[i][j][0] for j in range(len(neighbor_bond_dict[i]))])

        exp_left = []
        exp_right = []

        for i in range(radius):
            path_left = neighbor_list[left + i]
            path_right = neighbor_list[right - i]
            exp_right.append(atoms[max(path_left)])
            exp_left.append(atoms[min(path_right)])
        exp_left = exp_left[::-1]

        # local search of the atom list to find similar subgraph
        left_bound = exp_left + unit[:radius]
        right_bound = unit[-radius:] + exp_right

        for s in range(len(atoms)):
            if s != left or s != right:
                env_list = neighbor_list[s]
                if radius > 1:
                    for _ in range(1, radius):
                        env_list.append(neighbor_list[n] for n in env_list)
                        env_list = list(set(env_list))
                if len(list(set(atoms[env_list]) & set(left_bound))) <= 1:
                    subgraph[left] = subgraph[s]
                if len(list(set(atoms[env_list]) & set(right_bound))) <= 1:
                    subgraph[right] = subgraph[s]
        return subgraph


def transform_atoms(mol, atom_dict):
    """
    translation of atoms
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def extract_neighbor(mol, bond_dict):
    """
    find neighbors in the structure
    """
    neighbor_bond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        neighbor_bond_dict[i].append((j, bond))
        neighbor_bond_dict[j].append((i, bond))
    return neighbor_bond_dict


def split_dataset(dataset, ratio):
    """
    Shuffle and split a dataset.
    """
    np.random.seed(1234)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def data_generation(dataset, r, device):
    dir_dataset = './dataset/' + dataset + '/'
    # initialize different dicts before read data
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    subgraph_dict = defaultdict(lambda: len(subgraph_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    def create_dataset(filename):
        print(filename)
        with open(dir_dataset + filename, 'r') as f:
            smiles_property = f.readline().strip().split()
            data_original = f.read().strip().split('\n')
        data_original = [data for data in data_original
                         if '.' not in data.split()[0]]

        dataset = []

        for data in data_original:
            smiles, properties = data.strip().split()
            # mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            mol = Chem.MolFromSmiles(smiles)
            atoms = transform_atoms(mol, atom_dict)
            size = len(atoms)
            neighbor_bond_dict = extract_neighbor(mol, bond_dict)
            subgraph = torch.LongTensor(getsub(r, atoms, neighbor_bond_dict, subgraph_dict, edge_dict)).to(device)
            subgraph = add_pbc(atoms, r, neighbor_bond_dict, subgraph)
            adjmatrix = torch.FloatTensor(Chem.GetAdjacencyMatrix(mol)).to(device)
            dataset.append((subgraph, adjmatrix, size, torch.FloatTensor([float(properties)]).to(device)))

        return dataset

    dataset_train = create_dataset('data_train.txt')
    dataset_train, dataset_dev = split_dataset(dataset_train, 0.9)
    dataset_test = create_dataset('data_test.txt')

    N_subgraph = len(subgraph_dict)

    return dataset_train, dataset_dev, dataset_test, N_subgraph


class MolGNN(nn.Module):
    def __init__(self, num_sub, dimension, hidden_layer, output_layer):
        # initialization
        super(MolGNN, self).__init__()
        torch.manual_seed(42)
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        # the GNN is for updating subgraphs
        # GNN structure: 1st layer -> embedding
        self.embedding_subgraph = nn.Embedding(num_sub, dimension)
        self.w_subgraph = nn.ModuleList(nn.Linear(dimension, dimension) for _ in range(hidden_layer))
        self.w_out = nn.ModuleList(nn.Linear(dimension, dimension) for _ in range(output_layer))
        self.w_property = nn.Linear(dimension, 1)

    def pad(self, matrices, pad_value):
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i + m, j:j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.w_subgraph[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def gnn(self, inputs):
        # the GNN is used to embed subgraphs
        subgraph, adjmatrix, size = inputs  # data from func data_generation
        subgraph = torch.cat(subgraph)
        subgraph_vector = self.embedding_subgraph(subgraph)
        adjmatrix = self.pad(adjmatrix, 0)
        for l in range(self.hidden_layer):
            h = self.update(adjmatrix, subgraph_vector, l)
            subgraph_vector = torch.nn.functional.normalize(h, 2, 1)  # normalize

        # embedding vector is the summation of subgraph vectors
        molecule_vector = torch.stack([torch.sum(v, 0) for v in torch.split(subgraph_vector, size)])
        return molecule_vector

    def mlp(self, vectors):
        for l in range(self.output_layer):
            vectors = torch.relu(self.w_out[l](vectors))
        return self.w_property(vectors)

    def forward_regressor(self, data_batch, train):

        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])

        if train:
            molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate([correct_values], axis=0)
            return predicted_values, correct_values


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i + batch_train]))
            loss = self.model.forward_regressor(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total / N


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_regressor(self, dataset):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i + batch_test]))
            predicted_values, correct_values = self.model.forward_regressor(
                data_batch, train=False)
            SAE += sum(np.abs(predicted_values - correct_values))
        MAE = SAE / N
        return MAE

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')


def parse_cml_args(cml):
    arg = argparse.ArgumentParser(add_help=True)
    arg.add_argument('--dataset', dest='dataset', action='store', type=str,
                     choices=['polymer_i1', 'polymer_i2'],
                     default='polymer_i2',
                     help='dataset')
    arg.add_argument('--radius', dest='radius', action='store', type=int,
                     default=1,
                     help='radius to generate subgraphs')
    arg.add_argument('--dim', dest='dim', action='store', type=int,
                     default=25,
                     help='dimension of layers')
    arg.add_argument('--hidden', dest='layer_hidden', action='store', type=int,
                     default=5,
                     help='number of hidden layers')
    arg.add_argument('--output', dest='layer_output', action='store', type=int,
                     default=5,
                     help='number of output layers')
    arg.add_argument('--btrain', dest='batch_train', action='store', type=int,
                     default=256,
                     help='size of train batch')
    arg.add_argument('--btest', dest='batch_test', action='store', type=int,
                     default=128,
                     help='size of test batch')
    arg.add_argument('--lr', dest='lr', action='store', type=float,
                     default=1e-3,
                     help='learning rate')
    arg.add_argument('--lrdecay', dest='lr_decay', action='store', type=float,
                     default=0.99,
                     help='learning rate decay rate')
    arg.add_argument('--decayint', dest='decay_interval', action='store', type=int,
                     default=10,
                     help='decay interval')
    arg.add_argument('--iter', dest='iteration', action='store', type=int,
                     default=10,
                     help='number of iterations')
    return arg.parse_args(cml)


if __name__ == "__main__":

    arg = parse_cml_args(sys.argv[1:])

    (radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, decay_interval,
     iteration) = map(int, [arg.radius, arg.dim, arg.layer_hidden, arg.layer_output,
                            arg.batch_train, arg.batch_test,
                            arg.decay_interval, arg.iteration])
    lr, lr_decay = map(float, [arg.lr, arg.lr_decay])
    dataset = arg.dataset

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-' * 100)

    print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')
    (dataset_train, dataset_dev, dataset_test, N_subgraph) = data_generation(dataset, radius, device='cpu')
    print('-' * 100)

    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-' * 100)

    print('Creating a model.')
    torch.manual_seed(1234)

    model = MolGNN(
        N_subgraph, dim, layer_hidden, layer_output).to(device)
    print(model)
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-' * 100)

    file_result = './output/result.txt'
    result = 'Epoch\tTime(sec)\tLoss_train\tMAE_dev\tMAE_test'

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(1234)

    start = timeit.default_timer()

    for epoch in range(iteration):

        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)

        prediction_dev = tester.test_regressor(dataset_dev)
        prediction_test = tester.test_regressor(dataset_test)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-' * 100)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_train,
                                     prediction_dev, prediction_test]))
        tester.save_result(result, file_result)

        print(result)
