from rdkit import Chem
# import numpy as np
from rdkit.Chem import Draw
import deepchem as dc
from dgllife.data import MUV
import dgllife
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, ShuffleSplit
import numpy as np
from IPython.display import Image 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,normalize
from sklearn.metrics import r2_score,SCORERS,mean_absolute_error,max_error,mean_squared_error
from sklearn.model_selection import GridSearchCV
import argparse

# Initialize parser
parser = argparse.ArgumentParser()
 
# Data processing parameters
parser.add_argument("-data_size", "--data_size", help = "data size")
parser.add_argument("-test_ratio", "--test_ratio", help = "ratio of test set")
parser.add_argument("-val_ratio", "--val_ratio", help = "ratio of validation set")
data_size = 133000
test_ratio = 0.2
val_ratio = 0.25

# Training preprocessing parameters
parser.add_argument("-epochs", "--epochs", help = "number of epochs to train")
parser.add_argument("-batch_number", "--batch_number", help = "number of batches per epoch")
parser.add_argument("-learning_rate", "--learning_rate", help = "number of epochs to train")
parser.add_argument("-epochs", "--epochs", help = "number of epochs to train")
epochs = 1000
batch_size = 96
learning_rate = 0.003
 
 
# Read arguments from command line
args = parser.parse_args()

if args.data_size:
   data_size = args.data_size
   
if args.test_ratio:
   test_ratio = args.test_ratio

if args.val_ratio:
   val_ratio = args.val_ratio




def generate_atom_features(rdkit_mol,V):
  from rdkit import Chem,RDConfig
  from rdkit.Chem import rdPartialCharges
  import numpy as np
  import os
   
  #calculate acceptor or donor
  fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
  mol_featurizer = Chem.ChemicalFeatures.BuildFeatureFactory(fdef_name)
  mol_feats = mol_featurizer.GetFeaturesForMol(rdkit_mol)
  HB_list = ['Acceptor','Donor']
  HAD_list = []
  HAD_loc = []
  for n_feat,feat in enumerate(mol_feats):
    if feat.GetFamily() in HB_list:
        HAD_list.append(feat.GetAtomIds()[0])
        HAD_loc.append(n_feat)
  
  #calculate partial charge
  rdPartialCharges.ComputeGasteigerCharges(rdkit_mol)
  atoms_features = []
  for n,atom in enumerate(rdkit_mol.GetAtoms()):
    #1. Atom type
    atom_symbol_list = [
#         "B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S",
                        'C', 'N', 'O', 'S', 'F', 
#                         'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
#                          'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
#                          'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
#                          'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
                        'None']
    atom_symbol_features = np.zeros([1,len(atom_symbol_list)+1]).tolist()[0]

    if atom.GetSymbol() in atom_symbol_list:
      index = atom_symbol_list.index(atom.GetSymbol())
      atom_symbol_features[index] += 1
    # other atom symbol not in list
    else: 
      atom_symbol_features[-1] += 1

    #2. Chirality
    chirality_list = ['R','S']
    chirality_features = [0,0]
    try:
      chirality_features[chirality_list.index(atom.GetProp('_CIPCode'))] += 1
      chirality_features = chirality_features + atom.HasProp('_ChiralityPossible')
    except:
      chirality_features = chirality_features + [atom.HasProp('_ChiralityPossible')]

    #3. Formal charge:
    formal_charge = [atom.GetFormalCharge()]

    #4. Partial charge:
    partial_charge = [atom.GetDoubleProp('_GasteigerCharge')]

    #5. Is in ring size 3-8:
    ring_size_list = [3,4,5,6,7,8]
    ring_size_feature = [0,0,0,0,0,0]
    for size in range(len(ring_size_list)):
      if atom.IsInRingSize(ring_size_list[size]):
        ring_size_feature[size] += 1

    #6. Hybridization
    hybridization_list = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3]
    hybridization_feature = [0,0,0]
    if atom.GetHybridization() in hybridization_list:
      index = hybridization_list.index(atom.GetHybridization())
      hybridization_feature[index] += 1

    # #7. Hydrogen Bonding
    HB_list = ['Acceptor','Donor']
    HB_feature = [0,0]
    HAD = []
    if n in HAD_list:
        HAD = np.where(np.array(HAD_list)==n)[0]
    for HAD_a in HAD:
        if mol_feats[HAD_loc[HAD_a]].GetFamily() in HB_list:
          index = HB_list.index(mol_feats[HAD_loc[HAD_a]].GetFamily())
          HB_feature[index] += 1
    #8. Aromaticity
    aromaticity_feature = [0]
    if atom.GetIsAromatic():
      aromaticity_feature[0] += 1
    atom_features = atom_symbol_features+chirality_features+formal_charge+partial_charge+ring_size_feature+hybridization_feature+aromaticity_feature+HB_feature
#     atom_features = atom_symbol_features
    atoms_features.append(atom_features)
  if len(atoms_features)<V:
    for diff in range(V-len(atoms_features)):
        empty = np.zeros(24)
        empty[5]+=1
        atoms_features.append(empty)
  return np.array(atoms_features)
  
def max_pair_distance_pairs(mol,i,j,max_graph_distance):
  from rdkit import Chem
  from rdkit.Chem import rdmolops
  N = len(mol.GetAtoms())
  adj = rdmolops.GetAdjacencyMatrix(mol)
  higher_adj = np.linalg.matrix_power(adj, max_graph_distance)
  return higher_adj[i][j] == 1

def list2tensor(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float64)
    return arg

def generate_pair_features(rdkit_mol,max_graph_distance,V2):
  from rdkit import Chem
  import numpy as np
  N = len(rdkit_mol.GetAtoms()) #number of atoms
  pair_features = []
  A = Chem.rdmolops.GetAdjacencyMatrix(rdkit_mol) #adjacent matrix

  for i in range(N):
    for j in range(N):
      bond_ij = mol.GetBondBetweenAtoms(i,j)
      #1. bond type
      bond_type_list = ['SINGLE','DOUBLE','rdkit.Chem.rdchem.BondType.TRIPLE','rdkit.Chem.rdchem.BondType.AROMATIC']
      bond_type_feature = [0,0,0,0]
      if A[i,j] == 1:
        if str(bond_ij.GetBondType()) in bond_type_list:
          index = bond_type_list.index(str(bond_ij.GetBondType()))
          bond_type_feature[index] += 1

      #2. bond in ring
      ring_feature = [0]
      if A[i,j] == 1:
        if bond_ij.IsInRing():
          ring_feature[0] += 1

      #3. has pair within max_graph_distance:
      max_distance_feature = []
      for n in range(max_graph_distance):
        if max_pair_distance_pairs(rdkit_mol,i,j,n):
          max_distance_feature.append(1)
        else:
          max_distance_feature.append(0)
          
      pair_features.append(bond_type_feature+ring_feature+max_distance_feature)
  if len(pair_features)<V2:
    for diff in range(V2-len(pair_features)):
        empty = np.zeros(max_graph_distance+5)
        pair_features.append(empty)
  pair_features = np.array(pair_features).astype('float32')
  return pair_features

def GC_input_prepare(atom_features,pair_features,V):
  AP_ij_input = []
  for ai in range(len(atom_features)):
    for aj in range(len(atom_features)):
      AP_ij_input.append(atom_features[ai].tolist()+atom_features[aj].tolist())
  AP_ji_input = []
  for ai in range(len(atom_features)):
    for aj in range(len(atom_features)):
      AP_ji_input.append(atom_features[aj].tolist()+atom_features[ai].tolist())
  
  pair2atom = []
  for a in range(V):
    for b in range(V):
      pair2atom.append(a)
  PA_input = tf.math.segment_sum(pair_features,pair2atom)

  return atom_features,pair_features,np.array(AP_ij_input),np.array(AP_ji_input),np.array(PA_input)


class GraphConvLayer_1(keras.layers.Layer):
  def __init__(self,
               n_A_input = 26,
               n_P_input = 7,
               n_A_output = 50,
               n_P_output = 50,
               n_AA_depth = 50,
               n_AP_depth = 50,
               n_PP_depth = 50,
               n_PA_depth = 50,
               activation = 'relu',
               initializer = 'glorot_uniform'
               ):
    super().__init__()
    self.n_A_input = n_A_input
    self.n_P_input = n_P_input
    self.n_A_output = n_A_output
    self.n_P_output = n_P_output
    self.n_AA_depth = n_AA_depth
    self.n_AP_depth = n_AP_depth
    self.n_PP_depth = n_PP_depth
    self.n_PA_depth = n_PA_depth
    self.activataion = activation
    self.initializer = initializer

  def build(self, input_shape):

    #A->A
    self.w_AA = self.add_weight(shape=(self.n_A_input,self.n_AA_depth),
                                initializer=self.initializer,
                                trainable=True,name = 'wAA')
    self.b_AA = self.add_weight(shape=(self.n_AA_depth,),
                                initializer=self.initializer,
                                trainable=True,name = 'bAA')
    self.bn_AA = keras.layers.BatchNormalization(renorm = True,trainable = False)
    #P->P
    self.w_PP = self.add_weight(shape=(self.n_P_input,self.n_PP_depth),
                                initializer=self.initializer,
                                trainable=True,name = 'wPP')
    self.b_PP = self.add_weight(shape=(self.n_PP_depth,),
                                initializer=self.initializer,
                                trainable=True,name = 'bPP')
    self.bn_PP = keras.layers.BatchNormalization(renorm = True,trainable = False)

    #A->P P_ab = f(A_a,A_b)
    self.w_AP1 = self.add_weight(shape=(self.n_A_input*2,self.n_AP_depth),
                                initializer=self.initializer,
                                trainable=True,name = 'wAP1')
    self.b_AP1 = self.add_weight(shape=(self.n_AP_depth,),
                                initializer=self.initializer,
                                trainable=True,name = 'bAP1')
    self.w_AP2 = self.add_weight(shape=(self.n_A_input*2,self.n_AP_depth),
                                initializer=self.initializer,
                                trainable=True,name = 'wAP2')
    self.b_AP2 = self.add_weight(shape=(self.n_AP_depth,),
                                initializer=self.initializer,
                                trainable=True,name = 'bAP2') 
    self.bn_AP1 = keras.layers.BatchNormalization(renorm = True,trainable = False)
    self.bn_AP2 = keras.layers.BatchNormalization(renorm = True,trainable = False)


    
    #P->A
    self.w_PA = self.add_weight(shape=(self.n_P_input,self.n_PA_depth),
                                initializer=self.initializer,
                                trainable=True,name = 'wPA')
    self.b_PA = self.add_weight(shape=(self.n_PA_depth,),
                                initializer=self.initializer,
                                trainable=True,name = 'bPA')   
    self.bn_PA = keras.layers.BatchNormalization(renorm = True,trainable = False)

    #output A

    self.w_A_output = self.add_weight(shape=(self.n_PA_depth,self.n_A_output),
                                      initializer=self.initializer,
                                      trainable=True,name = 'wA') 
    self.b_A_output = self.add_weight(shape=(self.n_A_output,),
                                      initializer=self.initializer,
                                      trainable=True,name = 'bA') 
    self.bn_A = keras.layers.BatchNormalization(renorm = True,trainable = False)

    
    #output P
    self.w_P_output = self.add_weight(shape=(self.n_AP_depth,self.n_P_output),
                                      initializer=self.initializer,
                                      trainable=True,name = 'wP') 
    self.b_P_output = self.add_weight(shape=(self.n_P_output,),
                                      initializer=self.initializer,
                                      trainable=True,name = 'bP')  
    self.bn_P = keras.layers.BatchNormalization(renorm = True,trainable = False)

      
  def call(self, input):
    atom_input = input[0]
    pair_input = input[1]
    AP_ij_input = input[2]
    AP_ji_input = input[3]
    PA_input = input[4]
    activation = tf.keras.activations.get(self.activataion)

    AA = tf.matmul(atom_input,self.w_AA)+ self.b_AA
    AA = self.bn_AA(AA)
    AA = activation(AA)
    
    PP = tf.matmul(pair_input,self.w_PP) + self.b_PP
    PP = self.bn_PP(PP)
    PP = activation(PP)
    
    AP_ij = tf.matmul(AP_ij_input,self.w_AP1) + self.b_AP1
    AP_ij = self.bn_AP1(AP_ij)
    AP_ij = activation(AP_ij)
    AP_ji = tf.matmul(AP_ji_input,self.w_AP2) + self.b_AP2
    AP_ji = self.bn_AP2(AP_ji)
    AP_ji = activation(AP_ji)
    AP = AP_ij+AP_ji
    
    PA = tf.matmul(PA_input,self.w_PA) + self.b_PA
    PA = self.bn_PA(PA)
    PA = activation(PA)

    A = tf.matmul(AA+PA,self.w_A_output) + self.b_A_output
    A = self.bn_A(A)
    A = activation(A)
    P = tf.matmul(PP+AP,self.w_P_output) + self.b_P_output
#     P = tf.matmul(PP,self.w_P_output) + self.b_P_output
    P = self.bn_P(P)
    P = activation(P)


    return A,P
    
class GraphConvLayer_2(keras.layers.Layer):
  def __init__(self,
               n_A_input = 26,
               n_P_input = 7,
               n_A_output = 50,
               n_P_output = 50,
               n_AA_depth = 50,
               n_AP_depth = 50,
               n_PP_depth = 50,
               n_PA_depth = 50,
               activation = 'relu',
               initializer = 'glorot_uniform'
               ):
    super().__init__()
    self.n_A_input = n_A_input
    self.n_P_input = n_P_input
    self.n_A_output = n_A_output
    self.n_P_output = n_P_output
    self.n_AA_depth = n_AA_depth
    self.n_AP_depth = n_AP_depth
    self.n_PP_depth = n_PP_depth
    self.n_PA_depth = n_PA_depth
    self.activataion = activation
    self.initializer = initializer

  def build(self, input_shape):

    #A->A
    self.w_AA = self.add_weight(shape=(self.n_A_input,self.n_AA_depth),
                                initializer=self.initializer,
                                trainable=True,name = 'wAA')
    self.b_AA = self.add_weight(shape=(self.n_AA_depth,),
                                initializer=self.initializer,
                                trainable=True,name = 'bAA')
    self.bn_AA = keras.layers.BatchNormalization(renorm = True,trainable = False)
    #P->P
    self.w_PP = self.add_weight(shape=(self.n_P_input,self.n_PP_depth),
                                initializer=self.initializer,
                                trainable=True,name = 'wPP')
    self.b_PP = self.add_weight(shape=(self.n_PP_depth,),
                                initializer=self.initializer,
                                trainable=True,name = 'bPP')
    self.bn_PP = keras.layers.BatchNormalization(renorm = True,trainable = False)

    #A->P P_ab = f(A_a,A_b)
    self.w_AP1 = self.add_weight(shape=(self.n_A_input*2,self.n_AP_depth),
                                initializer=self.initializer,
                                trainable=True,name = 'wAP1')
    self.b_AP1 = self.add_weight(shape=(self.n_AP_depth,),
                                initializer=self.initializer,
                                trainable=True,name = 'bAP1')
    self.w_AP2 = self.add_weight(shape=(self.n_A_input*2,self.n_AP_depth),
                                initializer=self.initializer,
                                trainable=True,name = 'wAP2')
    self.b_AP2 = self.add_weight(shape=(self.n_AP_depth,),
                                initializer=self.initializer,
                                trainable=True,name = 'bAP2') 
    self.bn_AP1 = keras.layers.BatchNormalization(renorm = True,trainable = False)
    self.bn_AP2 = keras.layers.BatchNormalization(renorm = True,trainable = False)


    
    #P->A
    self.w_PA = self.add_weight(shape=(self.n_P_input,self.n_PA_depth),
                                initializer=self.initializer,
                                trainable=True,name = 'wPA')
    self.b_PA = self.add_weight(shape=(self.n_PA_depth,),
                                initializer=self.initializer,
                                trainable=True,name = 'bPA')   
    self.bn_PA = keras.layers.BatchNormalization(renorm = True,trainable = False)

    #output A

    self.w_A_output = self.add_weight(shape=(self.n_PA_depth,self.n_A_output),
                                      initializer=self.initializer,
                                      trainable=True,name = 'wA') 
    self.b_A_output = self.add_weight(shape=(self.n_A_output,),
                                      initializer=self.initializer,
                                      trainable=True,name = 'bA') 
    self.bn_A = keras.layers.BatchNormalization(renorm = True,trainable = False)

    
    #output P
    self.w_P_output = self.add_weight(shape=(self.n_AP_depth,self.n_P_output),
                                      initializer=self.initializer,
                                      trainable=True,name = 'wP') 
    self.b_P_output = self.add_weight(shape=(self.n_P_output,),
                                      initializer=self.initializer,
                                      trainable=True,name = 'bP')  
    self.bn_P = keras.layers.BatchNormalization(renorm = True,trainable = False)

      
  def call(self, input):
    atom_input = input[0]
    pair_input = input[1]
    AP_ij_input = input[2]
    AP_ji_input = input[3]
    PA_input = input[4]
    activation = tf.keras.activations.get(self.activataion)

    AA = tf.matmul(atom_input,self.w_AA)+ self.b_AA
    AA = self.bn_AA(AA)
    AA = activation(AA)
    
    PP = tf.matmul(pair_input,self.w_PP) + self.b_PP
    PP = self.bn_PP(PP)
    PP = activation(PP)
    
    AP_ij = tf.matmul(AP_ij_input,self.w_AP1) + self.b_AP1
    AP_ij = self.bn_AP1(AP_ij)
    AP_ij = activation(AP_ij)
    AP_ji = tf.matmul(AP_ji_input,self.w_AP2) + self.b_AP2
    AP_ji = self.bn_AP2(AP_ji)
    AP_ji = activation(AP_ji)
    AP = AP_ij+AP_ji
    
    PA = tf.matmul(PA_input,self.w_PA) + self.b_PA
    PA = self.bn_PA(PA)
    PA = activation(PA)

    A = tf.matmul(AA+PA,self.w_A_output) + self.b_A_output
    A = self.bn_A(A)
    A = activation(A)
    P = tf.matmul(PP+AP,self.w_P_output) + self.b_P_output
#     P = tf.matmul(PP,self.w_P_output) + self.b_P_output
    P = self.bn_P(P)
    P = activation(P)
    AP = tf.concat([A,P],1)

    return AP
    
class GraphPooling(keras.layers.Layer):
    
    def call(self,H):
        z = tf.reduce_sum(H,1)
        return z


from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor



dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True), return_smiles=True,
                                           target_index=np.random.choice(range(data_size), data_size, False))


V = 9
y = []
X = []
X1 = []
X2 = []
X3 = []
X4 = []
X5 = []

for smile in dataset_smiles:
    mol = Chem.MolFromSmiles('{}'.format(smile))
    atom_features = generate_atom_features(mol,V)
    pair_features = generate_pair_features(mol,2,V**2)
    x1,x2,x3,x4,x5 = GC_input_prepare(atom_features,pair_features,V)
    X1.append(x1)
    X2.append(x2)
    X3.append(x3)
    X4.append(x4)
    X5.append(x5)
    x = GC_input_prepare(atom_features,pair_features,V)
    X.append(x)

for data in dataset:
    data = data[2]
    y.append(data[5])
y = np.array(y)

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg
X1 = my_func(X1)
X2 = my_func(X2)
X3 = my_func(X3)
X4 = my_func(X4)
X5 = my_func(X5)

y_mm = MinMaxScaler().fit_transform(y.reshape(-1,1))

X1_train,X1_test,X2_train,X2_test,X3_train,X3_test,X4_train,X4_test,X5_train,X5_test,y_train,y_test = \
train_test_split(X1.numpy(),X2.numpy(),X3.numpy(),X4.numpy(),X5.numpy(),y_mm,test_size=test_ratio)

# 2 weave + pool
n_A_input = 24
n_P_input = 7
n_A_output = 50
n_P_output = 50
activation = 'relu'
initializer = 'glorot_uniform'
n_layer_size = [20,5]


#Input layer
A_input = keras.Input(name = 'A_input',shape=(9,n_A_input))
P_input = keras.Input(name = 'P_input',shape=(81,n_P_input))
Aij_input = keras.Input(name = 'Aij',shape=(81,2*n_A_input),dtype=tf.float32)
Aji_input = keras.Input(name = 'Aji',shape=(81,2*n_A_input,),dtype=tf.float32)
PA_input = keras.Input(name = 'PA', shape=(9,n_P_input),dtype=tf.float32)
inputs = [A_input,P_input,Aij_input,Aji_input,PA_input]

# Weave layer

weavelayer_1 = GraphConvLayer_1(n_A_input = n_A_input,
                          n_P_input = n_P_input,
                          n_A_output = n_A_input,
                          n_P_output = n_P_input,
                          activation = activation,
                          initializer = initializer)
weave_A, weave_P = weavelayer_1(inputs)
inputs = [weave_A,weave_P,Aij_input,Aji_input,PA_input]

weavelayer_2 = GraphConvLayer_2(n_A_input = n_A_input,
                          n_P_input = n_P_input,
                          n_A_output = n_A_output,
                          n_P_output = n_P_output,
                          activation = activation,
                          initializer = initializer)
weave_AP = weavelayer_2(inputs)


weave_gather = GraphPooling()
molecule_layer = weave_gather(weave_AP)

#Post reduction
layer1 = keras.layers.Dense(n_layer_size[0],activation = activation)(molecule_layer)
layer2 = keras.layers.Dense(n_layer_size[1],activation = activation)(layer1)

#Output layer
output = keras.layers.Dense(1)(layer2)
#   output = keras.layers.Reshape((n_task))(output)
#   output = keras.activations.softmax(output)


model = tf.keras.Model(inputs=[A_input,P_input,Aij_input, Aji_input, PA_input],
                  outputs=output)

print(model.summary())


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.MeanSquaredError(),
    # List of metrics to monitor
    metrics=[keras.metrics.MeanSquaredError()],
)

# use batch size of 50
# use 500 data as train and 4500 as validation
history=model.fit(
    x = [X1_train,X2_train,X3_train,X4_train,X5_train],
    y= y_train, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_split = val_ratio,
)

y_train_pred = model.predict(
    x = [X1_train,X2_train,X3_train,X4_train,X5_train],
    batch_size = batch_size,
)
y_test_pred = model.predict(
    x = [X1_test,X2_test,X3_test,X4_test,X5_test],
    batch_size =batch_size,
)

import matplotlib.pyplot as plt
import json
history_dict = history.history
json.dump(history_dict, open('history', 'w'))

model.save_weights('weight')

print(min(history.history['val_mean_squared_error']))
#plt.plot(history.history['mean_squared_error'])
#plt.plot(history.history['val_mean_squared_error'])
#plt.savefig('../fig/MSEvsEpoch_long')

#total = np.concatenate([y_test.reshape(data_size*test_ratio,1),y_test_pred])
#plt.scatter(y_test,y_test_pred)
#plt.plot((min(total),max(total)),(min(total),max(total)),color='r')
print('Test set R2 =',r2_score(y_test,y_test_pred))
print('Test set MSE =',mean_squared_error(y_test,y_test_pred))
print('Test set Max Error =',max_error(y_test,y_test_pred))
#plt.savefig('../fig/parity_test_long')

#total = np.concatenate([y_train.reshape(data_size*(1-test_ratio),1),y_train_pred])
#plt.scatter(y_train,y_train_pred)
#plt.plot((min(total),max(total)),(min(total),max(total)),color='r')
print('train set R2 =',r2_score(y_train,y_train_pred))
print('train set MSE =',mean_squared_error(y_train,y_train_pred))
print('train set Max Error =',max_error(y_train,y_train_pred))
#plt.savefig('../fig/parity_train_long')

