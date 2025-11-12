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
parser.add_argument("dcd_file", type=str, help="DCD trajectory file")
parser.add_argument("top_file", type=str, help="Topology file")
parser.add_argument("csv_file", type=str, help="csv file")
parser.add_argument("name_dataset", type=str, help="Name for grapgh dataset")
parser.add_argument("name_timelag", type=str, help="Name for the time-lag dataset")
parser.add_argument("cutoff", type=float, help="Cutoff")
parser.add_argument("gpu", type=str, help="gpu")
args = parser.parse_args()

device = torch.device(args.gpu)
torch.cuda.empty_cache()

dataset = dataset_from_conf(
    trajectory = args.dcd_file,
    top = args.top_file,
    cutoff = args.cutoff,  # Ang
    save = True,
    use_bonded_only = True,
    name_file = args.name_dataset,
    device = device
)
torch.save(dataset[:100000], './data/chunk_20ms.pt')

#####***** If you already have the graph dataset and want to create the time-lag dataset, uncomment next line *****#####
#dataset = torch.load('./data/heavy/bonds/villin_all.pt', map_location=device)


label_data = pd.read_csv(args.csv_file)
label_data['Ka'] = args.k_force * label_data['Ka']
label_data['Kb'] = args.k_force * label_data['Kb']
labels = [[torch.tensor(value, device=device) for value in sublist] for sublist in label_data[['Ka', 'Kb', 'center', 'weight']].to_numpy().tolist()]
datasets = create_timelagged_dataset(args.name_timelag, dataset, labels, lag_time=2, balance=0.75)
