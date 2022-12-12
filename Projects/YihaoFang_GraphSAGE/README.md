Fraud detection with GNNs
======================================

In this project, the inductive graph representation learning algorithm was applied to credit card transaction networks. Based on the transaction data, a hetergonerous tripatite graph was built up with $3$ different nodes as client, merchant and transactions nodes. GraphSAGE model was used as a feature aggregator from the graph, generating embeddings for transactions coming to the graph. Those embeddings were treated as additional features for the down stream task to determine whether the new transaction was fraudulent or not. Metrics from XGBoost indicated the significant lift from those additional embeddings. 

## Getting Started

### Installation

After downloading the code, you may install it by running

```bash
pip install -r requirements.txt
```

### Data

Data samples could be found as demo_ccf.csv in the data folder.


## Run

### Training

run

```bash
python3 GraphSAGE.py
```

to train the GraphSGAE model and generate the plot of PR curve.

