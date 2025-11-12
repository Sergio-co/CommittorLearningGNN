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
parser.add_argument("name", type=str, help="Name for the trained model")
parser.add_argument("dataset", type=str, help="time-lag dataset")
parser.add_argument("gpu", type=str, help="gpu")
args = parser.parse_args()
device = torch.device(args.gpu)
torch.cuda.empty_cache()


##********** This is to read the full graph dataset if you have it ready **************###
datasets = torch.load(args.dataset, map_location=device)
##***********************************************************************************##


gnn = GNNModel(
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

train_data, val_data = train_val_dataset(datasets, train_ratio=0.8)

model_name = f'./trained_models/{args.name}_k{args.k_force:.1f}'
epochs = args.epochs
patience = args.patience
batch_size = 700
lr = 1e-4
k = torch.tensor(args.k_force, device=device)
l2_lambda = 1e-5   #Regularization L2
l1_lambda = 1e-5   #Regularization L1


def init_weights1(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

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
    patience=patience, batch_size=batch_size, dataloader=DataLoader, lr=lr, kforce=k, wd=l2_lambda, wd1=l1_lambda)
    
