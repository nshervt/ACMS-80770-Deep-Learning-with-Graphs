## Recommender System with LightGCN model 


@University of Notre Dame
Created on Oct. 2022
Pytorch Implementation of LightGCN in [1] 
[1] Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jialu Wang (jwang44@nd.edu)


## Environment Requirement 


## Dataset 

Gowalla, Yelp2018

see more in dataloader.py 

## Run 

This code can run on terminal or on a provided jupyter notebook. 

Run on terminal:

* change base directory 

change `ROOT_PATH` in `world.py`

* command 

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64`

Run on Jupyter notebook:

* main.ipynb 

*NOTE*:

Change datasets: go to `parse.py`, change `--dataset`.

