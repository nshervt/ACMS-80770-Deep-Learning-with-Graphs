Optimization of Graph Neural Network for Predicting the Properties of Polymers
======================================

It is a common practice to use GNN to predict various properties of molecules. However, the performance of this method will be significantly reduced when facing larger molecules, especially polymers, which contains very long chains in their structures. This work develops an improved GNN model to address the challenge of limited accuracy in predicting polymer properties. In this method, the subgraph with radius r is used as the embedding object firstly, and then the description of the periodic boundary of the polymer units is optimized by combining the local searching methods and dynamic expanding of the molecular descriptions. The performance of the model on predicting the glass transition temperature of polymers is tested to be better than regular GNN model.

## Getting Started

### Dependencies

This implementation requires:

* Python (>= 3.7.3)
* SciPy (>= 1.7.3)
* PyParsing (>= 3.0.9)
* PyTorch (>= 1.10.2)
* RDKit (>= 2020.09.1.0)
* NumPy (>= 1.21.5)
* scikit-learn (>= 1.0.2)
* Matplotlib (>= 3.5.3)
* chainer-chemistry (>=0.7.1)

### Installation

After downloading the code, you may install it by running

```bash
pip install -r requirements.txt
```

### Data

There are two datasets are offered. For the dataset "polymer_i1", the structure is relatively simple, and this dataset is used for testing and debugging. "polymer_i2" dataset is a much larger dataset that from our groups researches. More polymers are included and some of their structures are very complex. Failure may occured when use our model to predict the properties of this dataset. Advanced structure of the model is needed.

## Run

```bash
python PolyGraph.py 
```

the argments are explained in the PolyGraph.py.
