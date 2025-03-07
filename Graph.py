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
import matplotlib.font_manager as font_manager
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from common.Models import GNNModel

parent_dir = os.path.abspath('../../')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
os.environ["PATH"] += os.pathsep + '/sbin'

def load_fon():
    font_path = os.path.join(parent_dir,'/home/scontreras/miniconda3/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')
    #font_path = os.path.join(parent_dir,'/home/sergiocontrerasarredondo/miniconda3/fonts/arial.ttf')
    assert os.path.exists(font_path)
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rc('font', family='sans-serif')
    matplotlib.rcParams.update({
        'font.size': 21,
        'font.sans-serif': prop.get_name(),
    })

load_fon()

plt.rcParams.update({
    "pgf.texsystem": "lualatex",
    #"font.family": "serif",  # use serif/main font for text elements
    "text.usetex": False,     # use inline math for ticks
    "mathtext.fontset": "cm",
    "axes.labelsize": 28,
    "axes.linewidth": 2.0,
    "font.size": 18,
    "axes.unicode_minus": False,
    "pgf.preamble": '\n'.join(["\\usepackage{units}"])
})

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
image_name = 'biased_k1.0.jpeg'
scripted_model = torch.load("./trained_models/biased_k10000.0_best_model.ptc", map_location=device)
scripted_model.eval()
dataset = torch.load('biased100ns300k.pt')
dataset = [data.to(device) for data in dataset]
data = pd.read_csv('./data/100ns300k_rmsd/rmsd2d.csv')#, comment='#', sep='\s+', header=None, names=['step','phi','psi'])
loader = DataLoader(dataset, batch_size=20000, shuffle=False)
outputs = []
with torch.no_grad():
    for batch in loader:
        output = scripted_model(batch.node_s, batch.node_v, batch.edge_index, batch.edge_attr, batch.edge_v, batch.batch)
        outputs.append(output.cpu())

outputs = torch.cat(outputs)
outputs = (outputs - outputs.min())/(outputs.max()-outputs.min())
outputs = outputs[:len(data)]
data['committor'] = outputs.squeeze().numpy()
data = data[['committor','phi','psi']]
#data.to_csv('committor_values.csv', index=False)

def plot_pmf(energy_landscape, save_figure=None):
    #pmf = pmf - np.min(pmf)
    colormap = matplotlib.colormaps.get_cmap('RdBu_r').copy()
    colormap.set_over(color='lightgrey')
    colormap.set_under(color='lightgrey')
    #w, h = figaspect(1/1.3)
    #fig = plt.figure(figsize=(w, h), constrained_layout=True)

    xi = energy_landscape['phi'].to_numpy()
    yi = energy_landscape['psi'].to_numpy()
    zi = energy_landscape['committor'].to_numpy()
    
    boundaries = np.linspace(0, 1, 100)
    norm = matplotlib.colors.BoundaryNorm(boundaries, ncolors=colormap.N, clip=True)

    cf = plt.scatter(xi, yi, c=zi, cmap=colormap, norm=norm, s=1.0)

    ax = plt.gca()
    #ax.set_facecolor('lightgrey')
    #ax.set_xlim(100, 160)
    #ax.set_ylim(5, 30)
    ax.set_xlabel(r'$\phi$',fontsize = 24)
    ax.set_ylabel(r'$\psi$',fontsize = 24)

    ax.tick_params(direction='in', which='major', length=6.0, width=1.0, top=True, right=True)
    ax.tick_params(direction='in', which='minor', length=3.0, width=1.0, top=True, right=True)
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #ax.xaxis.set_major_locator(plt.FixedLocator(np.linspace(100, 160, 7)))
    #ax.yaxis.set_major_locator(plt.FixedLocator(np.linspace(10, 30, 3)))
    ax.set_aspect('equal', adjustable='box')
    clb = plt.colorbar(cf, ticks=np.linspace(0, 1, 9))
    clb.ax.set_title(r'q', pad=10.0,fontsize = 14)
    cf.set_clim(0, 1)
    plt.savefig(save_figure, dpi=300, bbox_inches='tight', transparent=False)
    #plt.show()
    return

plot_pmf(data, save_figure=image_name)
#plot_pmf_smooth(data, save_figure='smooth.jpeg')
#plt.hist(data["committor"], bins=100, edgecolor='black')
#plt.xlabel("Committor")
#plt.ylabel("Frecuencia")
#plt.title("Histograma de Frecuencias")
#plt.savefig('hist.jpeg')
#plt.show()

