#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
import torch.nn as nn
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from scipy.spatial import distance_matrix
from torch_geometric.utils import to_undirected

def dataset_from_conf(trajectory, top, cutoff, save=True, name_file='dataset.pt'):
    u = mda.Universe(top, trajectory)
    graphs = []
    for ts in u.trajectory:
        graph = Get_Graph(u.atoms, cutoff)
        graphs.append(graph)
    if save:    
        torch.save(graphs, name_file)

    return graphs


def Get_Graph(frame, cutoff):
    #selected_atoms = [1, 2, 3, 5, 9, 13, 14, 15, 17, 19]
    selected_atoms = [0, 1, 2, 4, 8, 12, 13, 14, 16, 18]
    coords = frame.positions[selected_atoms]
    com = np.mean(coords, axis=0)

    #***Node features***#
    node_s = np.linalg.norm(coords - com, axis=1, keepdims=True)
    node_v = coords - com
    dist_matrix = distance_matrix(coords, coords)

    #***Edge features***#
    edge_index = []
    edge_attr = []
    edge_vectors = []

    num_atoms = len(selected_atoms)
    bonded_pairs = set(tuple(sorted((bond.indices[0], bond.indices[1]))) for bond in frame.bonds)
    bonded_pairs = {(i, j) for i, j in bonded_pairs if i in selected_atoms and j in selected_atoms}

    for local_i, i in enumerate(selected_atoms):
        for local_j, j in enumerate(selected_atoms):
            if local_i < local_j and dist_matrix[local_i, local_j] <= cutoff: #and (i, j) in bonded_pairs:
                edge_index.append([local_i, local_j])
                edge_index.append([local_j, local_i])  # Grafo no dirigido
                
                distance = dist_matrix[local_i, local_j]
                edge_attr.append([distance])
                edge_attr.append([distance])
                
                direction = (coords[local_j] - coords[local_i]) / distance
                edge_vectors.append(direction)
                edge_vectors.append(-direction)

    edge_index = torch.tensor(edge_index, dtype=torch.long).T if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float) if edge_attr else torch.empty((0, 1), dtype=torch.float)
    edge_vectors = torch.tensor(np.array(edge_vectors), dtype=torch.float) if edge_vectors else torch.empty((0, 3), dtype=torch.float)

    node_s = torch.tensor(node_s, dtype=torch.float)
    node_v = torch.tensor(node_v, dtype=torch.float)
    
    
    #++++++++++Uncomment if you want angles and dihedrals in the graphs+++++++++#
    #****** Angles and Dihedrals ******#
    '''
    angles = []
    dihedrals = []

    for angle in frame.angles:
        i, j, k = angle.indices
        vec1 = coords[i] - coords[j]
        vec2 = coords[k] - coords[j]
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angles.append(theta)
    
    for dihedral in frame.dihedrals:
        i, j, k, l = dihedral.indices
        vec1 = coords[i] - coords[j]
        vec2 = coords[k] - coords[j]
        vec3 = coords[l] - coords[k]
        n1 = np.cross(vec1, vec2)
        n2 = np.cross(vec2, vec3)
        m1 = np.cross(n1, vec2)
        
        x = np.dot(n1, n2)
        y = np.dot(m1, n2) * np.linalg.norm(vec2)
        phi = np.arctan2(y, x)
        dihedrals.append(phi)'''

    #******Crate the graph******#
    graph = Data(
        node_s=node_s,
        node_v=node_v,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_v=edge_vectors,
        #angles=torch.tensor(angles, dtype=torch.float),
        #dihedrals=torch.tensor(dihedrals, dtype=torch.float)
    )

    return graph

def create_timelagged_dataset(dataset, dataset1, lag_time=2):
    lag = int(lag_time)

    #**** This is to skip the first 25ns in the simulation ****#
    dataset = dataset[500000:-1]
    dataset1 = dataset1[500000:]

    label0 = dataset1[:-lag]
    label1 = dataset1[lag:]
    labels = [a + b for a, b in zip(label0, label1)]

    data0 = dataset[:-lag]
    data1 = dataset[lag:]
    tupla = [(data0[i], data1[i], labels[i]) for i in range(len(data0))]
    
    #** This is to save a shorter traj **#
    torch.save(tupla[:500000], 'Full_biased25ns.pt')
    torch.save(tupla[:100000], 'Full_biased5ns.pt')
    torch.save(tupla, 'Full_biased100ns.pt')
    
    return tupla
    
def train_val_dataset(dataset, train_ratio=0.8):
    num_graphs = len(dataset)
    num_train = int(num_graphs * train_ratio)
    num_val = num_graphs - num_train
    datasets = ListToDataset(dataset)
    train_dataset, val_dataset = random_split(datasets, [num_train, num_val])

    return train_dataset, val_dataset

class ListToDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    
 
