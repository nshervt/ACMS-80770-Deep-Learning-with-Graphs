Using Molecular Graph Convolution Neural Networks to Predict HOMO Energy in QM9 Dataset
======================================
Computer aided drug design has become an important approach in drug discovery. The properties of drug molecule candidates are manipulate and quantify with mathematical tool base on their structural features. With the advent of deep learning with graph, people started using graph to describe molecule where atoms represent nodes and bond represent edges. Many different graph based neural networks are proposed to as molecular level representations in machine learning applications.

In this work, I implemented the "Weave" representation proposed by Steven Kearnes. This weave representation consider molecules as graph and extract meaningful features from simple descriptions of the graph structure—atom and bond properties, and graph distances—to form molecule-level representations. Then I trained the implemented model with QM9 dataset to predict the HOMO energy.
## Getting Started

### Dependencies
* Python (= 3.10.6)
* NumPy (= 1.23.4)
* Matplotlib (= 3.5.3)
* Tensorflow (= 2.10.0)
* RDKit (= 2022.09.1)
* scikit-learn (= 0.24.2)
* chainer-chemistry (=0.7.1)

### Installation

After downloading the code, you may install it by running

```bash
pip install -r requirements.txt
```

## Run
Data generation, model construction, training and quantifying uncertainties are all in 'main.py'

### Data
```bash
optional arguments:
  -data_size           Total size of the training + test dataset (default: 133000)
  -test_ratio          Ratio of the test set. (default: 0.2)
  -val_ratio           Ratio of the validation set. (default: 0.25)
```



### Training
```bash
optional arguments:
  -epochs              number of epochs to train (default: 1000)
  -batch_number        number of batches per epoch (default: 96)
  -learning_rate       learning rate (default: 0.003)
```

After run 'main.py', a 'history' and 'weight' file will be generated. 'history' files contain the loss of training and validation set during training. 'weight' file contain trained parameters of model. To load weights, run
```
model.load_weight('weight')
```
