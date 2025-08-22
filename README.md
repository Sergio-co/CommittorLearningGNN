# Committor-graph neural network (qGNN) 

In this repository you can find the code associated with the committor-graph neural network (qGNN) presented in the article ''Atomic-level committor learning without collective variables,'' Sergio Contreras Arredondo _et al._ (2025). As well, we present the neccessary files for simulation reproducibility using [NAMD](https://www.ks.uiuc.edu/Research/namd/) molecular simulation package and together with [Colvars](https://github.com/Colvars/colvars) module for the enhance sampling.

---

## 1. System Requirements

- **Operating System**: Linux (Ubuntu 22.04), macOS (13.0), Windows 11  
- **Programming Language**: Python 3.10+  
- **Dependencies**:  
  - PyTorch (>= 2.1)  
  - PyTorch Geometric (>= 2.5)  
  - NumPy (>= 1.24)  
  - SciPy (>= 1.11)  
  - matplotlib (>= 3.7)

- **Versions tested**:  
  - Ubuntu 22.04, Python 3.10, CUDA 12.1

- **Hardware requirements**:  
  - Normal usage: standard desktop/laptop with CPU (≥ 16 GB RAM)  
  - Training: GPU (NVIDIA RTX 3080 or better recommended, 60 GB RAM)  

## 2. Installation Guide

1. Clone the repository:  
   git clone https://github.com/Sergio-co/CommittorLearningGNN.git gnn
   
   cd gnn

## 3. Demo (Running the model)

**To run the model**:
```
python run.py epochs patience k_force gpu dcd top csv graph model
```
Where:
- epochs: integer referring to the number of epochs for the training.
- patience: integer referring to the number of epochs to stop if there is no improvement in the loss.
- k_force: float referring to the k constant of the loss function that gives importance to the boundary conditions.
- gpu: string parameter. If there is cuda, write "cuda:N" with N a number; if not "cpu".
- dcd: dcd file for the trajectory.
- top: topology file for the molecule.
- csv: csv file in which the time-lagged trajectory is written and the weigths for reweigthing the committor time-correlation function.
- graph: string referring to the name you give to the file containing all the graphs corresponding to the trajectory images.
- model: string referring to the name you give to the qGNN model.

In ''tools'' folder, the file Graph.py can be used to plot the model into a plane of two selected coordinates or predefined collective variables for visualization purposes.

Additionally, the ''config_NAMD'' folder contains all the configuration files for NAMD program and Colvars module to reproduce the simulated systems.

**Expected output**:

- A trained model saved in outputs/model.pt

**Expected run time**: 
- ~25 minutes on a small system like NANMA using a 25ns long simulation.
