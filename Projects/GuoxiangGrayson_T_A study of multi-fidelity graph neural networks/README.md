A study of multi-fidelity graph neural networks
======================================

Enhancing the flexibility of traditional numerical schemes via the modern data-driven approaches has gain much attentions in many areas. In this project, we follow a recent work of building the multi-fidelity graph neural network model to assist the traditional finite element approximation. We make a few potential extensions to the current framework so a broader range of problems can be considered.

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

Training data is generated via the FEniCS project and stored in folders ```Canti2D/``` and ```Quarter2D/```. To generate new data, run the ```Data_prepare.py``` inside each folder to perform finite element computation and the solution will be saved in ```*/Sol/```. 

## Run

### Training

The model for the 2D cantilever example is trained by script ```parallel_training-canti2D.py```.
The model for the 2D quarter annular example is trained by script ```parallel_training-annular2D.py```.




### Quantifying uncertainties

To perform UQ analysis, use `test.py`. The `test.py` script accepts the following arguments:

```bash
optional arguments:
  --BB_samples          number of samples for uncertainty quantification (default: 0)
  --N                   number of training data (default: 600)
  --database            name of the training database (default: 'QM9')
```
