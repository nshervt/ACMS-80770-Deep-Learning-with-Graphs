My Project Title
======================================

A short abstract of the work.

## Getting Started

### Dependencies

This implementation requires:

* Python (>= 3.5)
* SciPy (>= 1.4.1)
* PyParsing (>= 1.1)
* PyTorch (>= 1.5.0)
* RDKit (>= 2019.09.3)
* NumPy (>= 1.18.1)
* Seaborn (>= 0.9.0)
* scikit-learn (>= 0.22.1)
* Matplotlib (>= 3.1.1)
* chainer-chemistry (>=0.6.0)

### Installation

After downloading the code, you may install it by running

```bash
pip install -r requirements.txt
```

### Data

Data samples are generated through `data.py`. The script accepts the following arguments:

```bash
optional arguments:
  --data_size           Total size of the training + test dataset (default: 100000)
  --N                   Size of the training set. (default: 600)
```

## Run

### Training

The model is trained using `main.py`. This code accepts the following arguments:

```bash
optional arguments:
  --epochs              number of epochs to train (default: 1900)
  --batch_number        number of batches per epoch (default: 25)
  --gpu_mode            accelerate the script using GPU (default: 1)
  --z_dim               latent space dimensionality (default: 30)
  --seed                random seed (default: 1400)
```

After generating the data, run

```bash
python3 main.py
```

to train the base model.

### Quantifying uncertainties

To perform UQ analysis, use `test.py`. The `test.py` script accepts the following arguments:

```bash
optional arguments:
  --BB_samples          number of samples for uncertainty quantification (default: 0)
  --N                   number of training data (default: 600)
  --database            name of the training database (default: 'QM9')
```
