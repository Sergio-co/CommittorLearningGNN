#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
import torch.nn as nn
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from Histogram.histogram import HistogramScalar

def determine_AB_functor(histogram_file):
    potential = HistogramScalar()
    with open(histogram_file, 'r') as f_input:
        potential.read_from_stream(f_input)
    potential.data = potential.data - np.min(potential)
    def determine_AB(pos):
        if pos[0] < -50.0 and pos[0] > -120.0 and pos[1] < 120.0 and pos[1] > 50.0:
            energy = potential[pos]
            if energy < 1.0:
                return 'A'
            else:
                return 'M'
        elif pos[0] > 0. and pos[0] < 120.0 and pos[1] > -150. and pos[1] < 0.:
            energy = potential[pos]
            if energy < 4.0:
                return 'B'
            else:
                return 'M'
        else:
            return 'M'
    return determine_AB

energy_landscape_file = 'potential.dat'
determine_AB = determine_AB_functor(energy_landscape_file)

traj = pd.read_csv('../nanma/output2/unbiased_6ns600k.colvars.traj', comment='#', sep='\s+', header=None, names=['step','phi','psi'])
positions = traj[['phi', 'psi']].to_numpy()

traj['state'] = np.apply_along_axis(determine_AB, 1, positions)
#traj['label'] = -100
#traj.loc[traj['state'] == 'A', 'label'] = 0
#traj.loc[traj['state'] == 'B', 'label'] = 1
#traj.loc[traj['state'] == 'M', 'label'] = -1
traj['center'] = -1
traj.loc[traj['state'] == 'A', 'center'] = 0.0
traj.loc[traj['state'] == 'B', 'center'] = 1.0
traj['ka'] = 0.0
traj.loc[traj['state'] == 'A', 'ka'] = 10.0
traj['kb'] = 0.0
traj.loc[traj['state'] == 'B', 'kb'] = 10.0
if 'state' in traj.columns:
    traj.drop(['state'], axis=1, inplace=True)
traj.to_csv('basin6ns_k10.csv', index=False)
