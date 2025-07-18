# Committor-graph neural network (qGNN) 

In this repository you can find the code associated with the committor-graph neural network (qGNN) presented in the article ''Atomic-level committor learning without collective variables,'' Sergio Contreras Arredondo _et al._ (2025). As well, we present the neccessary files for simulation reproducibility using [NAMD](https://www.ks.uiuc.edu/Research/namd/) molecular simulation package and together with Colvars module for the enhance sampling.

## Running the model

To run the model:
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
