Topological connection between zeolites and their potential interzeolite transformation
======================================

Zeolites have a high industrial value due to their multiple applications in the adsorption of compounds or as catalysts.However, obtaining large zeolites is highly expensive, which is why the interconversion between zeolites is critical for the viability of these processes. This interconversion does not define descriptors that allow relating the zeolites, for which Variational Graph Autoencoding was applied to combine the relationship between the different atoms that make up the zeolite and the different descriptors that it is possible to extract from these atoms.The corresponding latent variable was used to cluster the different graphs that represent the different zeolites, obtaining good interconversion predictions for different zeolites, such as BEA and CHA and it is possible to extend this analysis to other organized structures such as the Metal–organic frameworks (MOFs)

## Getting Started

### Dependencies

This implementation requires:

* Python (>= 3.9)
* SciPy (>= 1.9.3)
* PyTorch (>= 1.13)
* NumPy (>= 1.23.4)
* Seaborn (>= 0.12.1)
* scikit-learn (>= 1.1.2)
* Matplotlib (>= 3.1.1)
* chainer-chemistry (>=0.6.0)
* dscribe (>= 1.2)
* csv (>= 1.0)
* ase (>= 3.22.1)
* pandas (>= 1.5.1)

### Installation

After downloading the code, you may install it by running

```bash
pip install -r requirements.txt
```

### Data

Data samples are generated through `GenerateDataset_for_GNN.py`. By the following line:

```bash
python3 GenerateDataset_for_GNN.py
```

## Run

### Training

The model is trained using `Training_VGAE_Zeolite.py`, by the following line:

```bash
python3 Training_VGAE_Zeolite.py
```

to train the base model.

### To Visualize and PCA analysis

To visualize and PCA analysis run the following command:

```bash
python3 KClustering_Analysis_wPCA.py
```
