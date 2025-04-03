#!/usr/bin/env python3
import argparse
import torch.nn as nn
import torch
import inspect
import os, sys
import pandas as pd
import numpy as np
import importlib.util
import gzip
import subprocess
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.figure import figaspect
from torch.utils.data import random_split
from common.Models import GNNModel
from common.train import train_model
from common.train import train_one_epoch
from common.committor_loss import loss, JAB
from torch_geometric.loader import DataLoader
from common.process_graph import dataset_from_conf, create_timelagged_dataset, train_val_dataset

parent_dir = os.path.abspath('../../')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
os.environ["PATH"] += os.pathsep + '/sbin'

parser = argparse.ArgumentParser(description="Run GNN model with specified parameters.")
parser.add_argument("epochs", type=int, help="Number of epochs")
parser.add_argument("patience", type=int, help="Patience")
parser.add_argument("k_force", type=float, help="Basin force constant")
parser.add_argument("gpu", type=str, help="gpu")
args = parser.parse_args()

#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.device(args.gpu)
torch.cuda.empty_cache()
gnn = GNNModel(
    n_cvs = 1,
    gnn_out = 1,
    node_s_dim = 16,
    node_v_dim = 8,
    edge_s_dim = 8,
    edge_v_dim = 6,
    n_layers = 4,
    n_messages = 3,
    n_feedforwards = 2,
    drop_rate = 0.0,
    activation = 'Tanh',
)

dataset = dataset_from_conf(
    trajectory='./data/March.11.GNN.data.Trialanine/rmsd.dcd',
    top='./data/March.11.GNN.data.Trialanine/trialanine.parm7',
    cutoff=20,  # Ang
    save=True,
    name_file='./data/bond_cutoff20/biased100ns300k.pt'       
)
#dataset = torch.load('./data/bond_cutoff20/biased100ns300k.pt')
dataset = [data.to(device) for data in dataset]
label_data = pd.read_csv('./data/March.11.GNN.data.Trialanine/tri-rmsd-2d.csv')
labels = [[torch.tensor(value, device=device) for value in sublist] for sublist in label_data[['Ka', 'Kb', 'center', 'weight']].to_numpy().tolist()]
datasets = create_timelagged_dataset(dataset, labels, lag_time=2, balance=0.7)

##*** This to read the full graph dataset if you have it ready ***###
#datasets = torch.load('./data/all/Full_lcombo_all.pt', map_location=device)
#datasets = [
#    (g1.to(device), g2.to(device), [t.to(device) for t in tensor_list]) 
#    for g1, g2, tensor_list in datasets
#]
##*****************************************************************##

train_data, val_data = train_val_dataset(datasets, train_ratio=0.8)

model_name = f'./trained_models/Trialanine_k{args.k_force:.1f}'
epochs = args.epochs
patience = args.patience
kforce = torch.tensor(args.k_force, device=device)
batch_size = 5000
lr = 1e-4
weight_decay = 1e-5

#/**** This is good for ReLU ****/
def init_weights1(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
#/**** This is good for Tanh ****/
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

gnn.to(device)
gnn.apply(init_weights)

best_model = train_model(
    model_to_train=gnn, output_prefix=model_name,
    train_set=train_data, val_set=val_data, loss_function=loss, epochs=epochs,
    patience=patience, batch_size=batch_size, dataloader=DataLoader, lr=lr, kforce=kforce, wd=weight_decay)
    

