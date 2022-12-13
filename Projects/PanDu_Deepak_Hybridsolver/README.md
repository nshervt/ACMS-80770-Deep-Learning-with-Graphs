CFD-GCN hybrid solver project
======================================

The project creates a hybrid solver that predicts hemodynamic flow information given random inlet profiles for a 2-D arotic flow case.

## Getting Started

### Dependencies

This implementation requires:

* Python (= 3.7)
* SciPy (>= 1.4.1)
* PyParsing (>= 1.1)
* PyTorch (= 1.5.0)
* NumPy (>= 1.18.1)
* Matplotlib (>= 3.1.1)
* torch-scatter==2.0.5
* torch-sparse==0.6.6
* torch-cluster==1.5.5
* torch-spline-conv==1.2.0
* torch-geometric==1.6.0 
* swig (>=3.0)
* mpicc
* mpi4py
* cudatoolkit=10.1
### Installation

After downloading the code, you may install it by running

```bash
###########install pytorch and pytorch geometric########################################
#create conda enviroment
conda create -n cfd-gcn python=3.7
conda activate cfd-gcn 
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
conda install pyg -c pyg
#Install torch_geometric
env PATH=/usr/local/cuda/bin:$PATH
env CPATH=/usr/local/cuda/include:$CPATH
env LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
pip install torch-sparse==0.6.6 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
pip install torch-cluster==1.5.5 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html
pip install torch-geometric==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html

###########install SU2########################################
#assuming your in the current path = your-own-path
apt-get update -y && apt-get install -y openmpi-bin libopenmpi-dev swig m4
env MPICC=/usr/bin/mpicc pip install mpi4py
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CXXFLAGS="-O3"
git clone --branch feature_pytorch_communicator  https://github.com/su2code/SU2 
mkdir su2install
cd SU2 
./preconfigure.py --enable-mpi --with-cc=/usr/bin/mpicc --with-cxx=/usr/bin/mpicxx --prefix=your-own-path/su2install/SU2 --enable-autodiff --enable-PY_WRAPPER --disable-tecio --update
sudo make -j 16 Install
#add this to ~/.bashrc
export PATH=$PATH:your-own-path/su2install/SU2/bin
export PYTHONPATH=$PYTHONPATH:your-own-path/su2install/SU2/bin
export SU2_RUN=your-own-path/su2install/SU2/bin
export SU2_HOME=your-own-path/SU2
```

### Data
After download the project code, you need to download the data folder from google drivelink:
https://drive.google.com/file/d/1VTnuxaQXppp2mKpLFD9KJ86RL1zCWYQf/view?usp=sharing

Inside the data folder:
Data samples are generated through `./data/cases/data_generation.py` The script accepts the following arguments: no arguments
num_cases = 2000 represent how many data you wanna generate and is hard coded inside the python file. 


and then through `./data.py`. The script accepts the following arguments:
```bash
--root    root directory of the cases files 
-d 	  output directory of the graph dataset files 
-fm       fine mesh file name for the graph dataset 
-ts       total size of the graph dataset

```

#Run this to generate the data 
```bash
conda activate cfd-gcn
cd ./data/cases
python data_generation.py > log
cd ..
cd ..
python data.py --root './data/cases' -d './data/dataset0' -fm './data/aorta3.su2' -ts 2000
```

## Run

### Training

The model is trained using `./train.py`. This code accepts the following arguments: 
```bash
-np   number of process for the training step. should batchsize+1
```

After generating the data, run

```bash
mpirun -np 5 python train.py > log
```

to train the base model.

### Plotting

run this to visualize the predictions :
```bash
python plot.py
```




