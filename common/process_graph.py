#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
import torch.nn as nn
import random
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from scipy.spatial import distance_matrix
from torch_geometric.utils import to_undirected
from tqdm import tqdm

def dataset_from_conf(trajectory, top, cutoff, use_bonded_only=False,
                      save=True, name_file='dataset.pt', device='cpu'):
    """
    Build a graph dataset from MD trajectory

    Args:
        trajectory (str): Trajectory file (e.g. .xtc, .dcd, etc.)
        top (str): Topology file (e.g. .pdb, .psf, .prmtop, etc.)
        cutoff (float): Cutoff for edges (Å).
        use_bonded_only (bool): if True, edges only between bonded atoms.
        save (bool): if True, save dataset to the disk.
        name_file (str): name of the dataset to save.
        device (str): 'cpu' or 'cuda' (GPU).

    Returns:
        list[torch_geometric.data.Data]: list of graphs.
    """
    u = mda.Universe(top, trajectory)
    graphs = []
    
    selected_atoms = u.atoms.select_atoms("not name H*").indices
    bonded_pairs = set(tuple(sorted((b.indices[0], b.indices[1]))) for b in u.bonds)
    bonded_pairs = {(i, j) for i, j in bonded_pairs if i in selected_atoms and j in selected_atoms}
        
    for ts in tqdm(u.trajectory, desc="Processing trajectory"):
        graph = Get_Graph(u.atoms, cutoff=cutoff,
                          use_bonded_only=use_bonded_only,
                          device=device,
                          selected_atoms=selected_atoms,
                          bonded_pairs=bonded_pairs)
        graphs.append(graph)

    if save:
        torch.save(graphs, name_file)

    return graphs

def Get_Graph(frame, cutoff=5.0, use_bonded_only=False, device='cpu', selected_atoms=None, bonded_pairs=None):
    """
    Build a graph from a single frame of the trajectory.

    Args:
        frame: frame of MDAnalysis.
        cutoff (float): Max distance for edges (Å).
        use_bonded_only (bool): if True, edges only between bonded atoms.
        device (str): 'cpu' o 'cuda'.
    """

    #selected_atoms = frame.select_atoms("not name H*").indices
    coords_np = frame.positions[selected_atoms]

    # --- coordinates to tensors ---
    coords = torch.as_tensor(coords_np, dtype=torch.float32, device=device)
    coords.requires_grad_(True)
    com = torch.mean(coords, dim=0, keepdim=True)

    # --- Node features ---
    node_s = torch.norm(coords - com, dim=1, keepdim=True)
    node_v = coords - com
    num_atoms = len(selected_atoms)

    # --- edges ---
    if not use_bonded_only:
        #  cutoff
        dist_matrix = torch.cdist(coords, coords, p=2)
        mask = (dist_matrix <= cutoff) & (torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool())
        i_idx, j_idx = torch.where(mask)

    else:
        global_to_local = {a: idx for idx, a in enumerate(selected_atoms)}
        pairs_local = torch.tensor(
            [[global_to_local[i], global_to_local[j]] for i, j in bonded_pairs],
            dtype=torch.long,
            device=device
        )

        if pairs_local.numel() > 0:
            i_idx, j_idx = pairs_local[:, 0], pairs_local[:, 1]
            diff = coords[j_idx] - coords[i_idx]
            dist = torch.norm(diff, dim=1)
            mask = dist <= cutoff
            i_idx, j_idx = i_idx[mask], j_idx[mask]
        else:
            i_idx = j_idx = torch.empty(0, dtype=torch.long, device=device)


    if len(i_idx) > 0:
        diff = coords[j_idx] - coords[i_idx]
        dist = torch.norm(diff, dim=1)
        directions = diff / dist.unsqueeze(1)

        edge_index = torch.cat([torch.stack([i_idx, j_idx]), torch.stack([j_idx, i_idx])], dim=1)
        edge_attr = torch.cat([dist, dist]).unsqueeze(1)
        edge_vectors = torch.cat([directions, -directions], dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), dtype=torch.float, device=device)
        edge_vectors = torch.empty((0, 3), dtype=torch.float, device=device)

    # --- Create graph ---
    graph = Data(
        node_s=node_s,
        node_v=node_v,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_v=edge_vectors,
    )

    return graph

def create_timelagged_dataset(name, dataset, dataset1, lag_time=2, balance=None):
    lag = int(lag_time)

    # balance --> points outside basins / total points
    #dataset = dataset[::10]
    #dataset1 = dataset1[::10]

    label0 = dataset1[:-lag]
    label1 = dataset1[lag:]
    weights = torch.sqrt(torch.abs(torch.tensor([row0[3] * row1[3] for row0, row1 in zip(label0, label1)], device=label0[0][0].device)))
    labels = [a + b + [c] for a, b, c in zip(label0, label1, weights)]

    data0 = dataset[:-lag]
    data1 = dataset[lag:]
    tupla = list(zip(data0, data1, labels))
    
    if balance:
        positive_samples = [t for t in tupla if t[2][2] == 0 or t[2][2] == 1]  # center = 0 o 1
        negative_samples = [t for t in tupla if t[2][2] == -1]  # center = -1

        print('********************************')
        print(f'Inside basin: {len(positive_samples)}')
        print(f'Outside basin: {len(negative_samples)}')
        print('********************************')


        target_negative_count = int(len(positive_samples) * balance / (1.0 - balance))
        if len(negative_samples) > target_negative_count:
            negative_samples = random.sample(negative_samples, target_negative_count)

        balanced_dataset = positive_samples + negative_samples
        random.shuffle(balanced_dataset)
        tupla = balanced_dataset

    random.shuffle(tupla)
    torch.save(tupla[:100000], name)
    
    return tupla
    
def renormalize_weights(weights):
    return weights / weights.sum() * weights.shape[0]

def apply_renormalization(dataset):
    device = next(iter(dataset))[0].edge_index.device
    weights = torch.tensor([tup[2][-1] for tup in dataset], dtype=torch.float32, device=device)
    new_weights = renormalize_weights(weights)
    new_dataset = [(tup[0], tup[1], tup[2][:-1] + [new_w]) for tup, new_w in zip(dataset, new_weights)] 
    return ListToDataset(new_dataset)

def train_val_dataset(dataset, train_ratio=0.8):
    num_graphs = len(dataset)
    num_train = int(num_graphs * train_ratio)
    num_val = num_graphs - num_train
    datasets = ListToDataset(dataset)
    train_dataset, val_dataset = random_split(datasets, [num_train, num_val])
   
    train_dataset = apply_renormalization(train_dataset)
    val_dataset = apply_renormalization(val_dataset)

    return train_dataset, val_dataset

class ListToDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    
