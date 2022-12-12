import sys
sys.path.append('./Project')
import pandas as pd 
import numpy as np
# import the graph models from utils folder
from utils.graphconstruction import GraphConstruction
from utils.hinsage import HinSAGE_Representation_Learner
from utils.evaluation import Evaluation
from xgboost import XGBClassifier

# Global parameters:
embedding_size = 64
add_additional_data = True

# read the data at ./data/
df = pd.read_csv("./data/demo_ccf.csv")
# first 60% of transactions are used for training, last 40% to test inductive capability
cutoff = round(0.6*len(df)) 
train_data = df.head(cutoff)
inductive_data = df.tail(len(df)-cutoff)

# build up the training graph
transaction_node_data = train_data.drop("client_node", axis=1).drop("merchant_node", axis=1).drop("fraud_label", axis=1).drop('index', axis=1)
client_node_data = pd.DataFrame([1]*len(train_data.client_node.unique())).set_index(train_data.client_node.unique())
merchant_node_data = pd.DataFrame([1]*len(train_data.merchant_node.unique())).set_index(train_data.merchant_node.unique())

nodes = {"client":train_data.client_node, "merchant":train_data.merchant_node, "transaction":train_data.index}
edges = [zip(train_data.client_node, train_data.index),zip(train_data.merchant_node, train_data.index)]
features = {"transaction": transaction_node_data, 'client': client_node_data, 'merchant': merchant_node_data}

graph = GraphConstruction(nodes, edges, features)
S = graph.get_stellargraph()

# GraphSAGE
num_samples = [2,32]
embedding_node_type = "transaction"

hinsage = HinSAGE_Representation_Learner(embedding_size, num_samples, embedding_node_type)
trained_hinsage_model, train_emb = hinsage.train_hinsage(S, list(train_data.index), train_data['fraud_label'], batch_size=5, epochs=10)

# inductive graph, includes all data
pd.options.mode.chained_assignment = None

train_data['index'] = train_data.index
inductive_data['index'] = inductive_data.index
inductive_graph_data = pd.concat((train_data,inductive_data))
inductive_graph_data = inductive_graph_data.set_index(inductive_graph_data['index']).drop("index",axis = 1)
transaction_node_data = inductive_graph_data.drop("client_node", axis=1).drop("merchant_node", axis=1).drop("fraud_label", axis=1)
client_node_data = pd.DataFrame([1]*len(inductive_graph_data.client_node.unique())).set_index(inductive_graph_data.client_node.unique())
merchant_node_data = pd.DataFrame([1]*len(inductive_graph_data.merchant_node.unique())).set_index(inductive_graph_data.merchant_node.unique())

nodes = {"client":inductive_graph_data.client_node, "merchant":inductive_graph_data.merchant_node, "transaction":inductive_graph_data.index}
edges = [zip(inductive_graph_data.client_node, inductive_graph_data.index),zip(inductive_graph_data.merchant_node, inductive_graph_data.index)]
features = {"transaction": transaction_node_data, 'client': client_node_data, 'merchant': merchant_node_data}

graph = GraphConstruction(nodes, edges, features)
S = graph.get_stellargraph()

# obtain embeddings for inductive graph
inductive_emb = hinsage.inductive_step_hinsage(S, trained_hinsage_model, inductive_data.index, batch_size=5)

# apply XGBoost on both original/original+embeddings 
classifier = XGBClassifier(n_estimators=100)

train_labels = train_data['fraud_label']

if add_additional_data is True:
    train_emb = pd.merge(train_emb, train_data.loc[train_emb.index].drop('fraud_label', axis=1), left_index=True, right_index=True)
    inductive_emb = pd.merge(inductive_emb, inductive_data.loc[inductive_emb.index].drop('fraud_label', axis=1), left_index=True, right_index=True)

    baseline_train = train_data.drop('fraud_label', axis=1)
    baseline_inductive = inductive_data.drop('fraud_label', axis=1)

    classifier.fit(baseline_train, train_labels)
    baseline_predictions = classifier.predict_proba(baseline_inductive)
    
classifier.fit(train_emb, train_labels)
predictions = classifier.predict_proba(inductive_emb)

inductive_labels = df.loc[inductive_emb.index]['fraud_label']

graphsage_evaluation = Evaluation(predictions, inductive_labels, "GraphSAGE+features") 
graphsage_evaluation.pr_curve()

if add_additional_data is True:
    baseline_evaluation = Evaluation(baseline_predictions, inductive_labels, "Baseline")
    baseline_evaluation.pr_curve()
predictions = classifier.predict_proba(inductive_emb)


