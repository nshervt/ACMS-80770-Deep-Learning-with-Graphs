A study of multi-fidelity graph neural networks
======================================

Enhancing the flexibility of traditional numerical schemes via the modern data-driven approaches has gain much attentions in many areas. In this project, we follow a [recent work](https://www.sciencedirect.com/science/article/abs/pii/S004578252200305X) of building the multi-fidelity graph neural network model to assist the traditional finite element approximation. We make a few potential extensions to the current framework so a broader range of problems can be considered.

## Preparation

### Dependencies

This implementation requires:

* Python (>= 3.5)
* SciPy (>= 1.4.1)
* PyTorch (>= 1.5.0)
* NumPy (>= 1.18.1)
* Matplotlib (>= 3.1.1)
* FEniCS (2019.1.0)
* Mpi4py (>= 3.1.4)
* Meshio (>= 2.3.5)
* Mgmetis (>=0.1.1)

### Data

Training data was generated via the FEniCS project and stored in folders ```Canti2D/``` and ```Quarter2D/```. To generate new data, run ```*/Data_prepare.py``` inside each folder to perform finite element computation and the `.vtu` solutions will be saved inside ```*/Sol/```. Then, run `*/mesh-part.py` to partition the computational mesh in parallel and the partitioned mesh will be saved as the `.vtk` file to the folder `*/Partitioned_Mesh/`. To use the [CRC](https://crc.nd.edu/) platform of university of Notre Dame, use the shell script `*/*-partitioning.script` as:
-  ```qsub *-partitioning.script```


## Run

### Training

The model for the 2D cantilever example is trained by the script ```parallel_training-canti2D.py```.

The model for the 2D quarter annular example is trained by the script ```parallel_training-annular2D.py```.

Current training script allows a grid search of the mini-batch size $n_B$ and the learning rate $\eta$. Other model parameters, hyperparameters can be accessed and changed in `Tools/Model.py`. The trained models are saved to the folder `Model_save/*/`. To use the [CRC](https://crc.nd.edu/) platform of university of Notre Dame, use the shell script `*_training.script` as:
-  ```qsub *_training.script```



### Model evaluation

At the current stage, only one model evaluation task can be done. Run the `Accuracy_evaluation.py` to generate histograms about accuracy recovery from the low-fidelity simulation to the high-fidelity simulation. The low-fidelity simulation corresponds to the finite element approximation on a coarse computational mesh and the high-fidelity simulation, associated to a dense mesh, is predicted via the multi-fidelity graph neural network. Change the argument:
- `prefix = Model/*/`
- `num_part = *`

to switch between two cases, i.e. the 2D cantilever problem partitioned to 128 parts and the 2D anuular problem partitioned to 256 parts.
