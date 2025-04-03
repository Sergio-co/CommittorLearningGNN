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
#from common.Models import GNNModel
from common.process_graph import ListToDataset
import matplotlib.colors
import matplotlib.tri as tri
from scipy.interpolate import griddata
from scipy.interpolate import splprep, splev
from numpy.polynomial.polynomial import Polynomial

parent_dir = os.path.abspath('../../')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
os.environ["PATH"] += os.pathsep + '/sbin'

parser = argparse.ArgumentParser(description="Run GNN model with specified parameters.")
parser.add_argument("k_force", type=float, help="Basin force constant")
parser.add_argument("gpu", type=str, help="gpu")
args = parser.parse_args()

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

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device(args.gpu)
name = f'./Images/all/combo3_w_k{args.k_force:.1f}.jpeg'
scripted_model = torch.load(f'./trained_models/all/combo3_w_k{args.k_force:.1f}_best_model.ptc', map_location=device)
#scripted_model.to(device)
scripted_model.eval()
pmf = pd.read_csv('./data/QMMM.all.czar.pmf', delimiter=r'\s+', comment='#', header=None)
pmf.columns = ['d1', 'd2', 'energy']

dataset = torch.load('./data/all/combo5ns_all.pt')
dataset = [data.to(device) for data in dataset]
data = pd.read_csv('./data/DA_RMSD_k1.csv.gz')#, comment='#', sep='\s+', header=None, names=['step','phi','psi'])

if len(dataset) < len(data):
    data = data.head(len(dataset))
elif len(dataset) >= len(data):
    dataset = dataset[:len(data)]

#dataset = dataset[63000:]
#data =  data.iloc[63000:]

dataset = ListToDataset(dataset)
loader = DataLoader(dataset, batch_size=20000, shuffle=False)
outputs = []
with torch.no_grad():
    for batch in loader:
        output = scripted_model(batch.node_s, batch.node_v, batch.edge_index, batch.edge_attr, batch.edge_v, batch.batch)
        #output = torch.clamp(output, min=0.0, max=1.0)
        outputs.append(output.cpu())

outputs = torch.cat(outputs)
#outputs = (outputs - outputs.min())/(outputs.max()-outputs.min())
#outputs = outputs[:len(data)]
data['committor'] = outputs.squeeze().numpy()
data = data[['committor','d1','d2']]
#data = data.sample(n=50000)
#data.to_csv('committor_values.csv', index=False)

def plot_pmf(energy_landscape, pmf, save_figure=None):
    #pmf = pmf - np.min(pmf)
    colormap = matplotlib.colormaps.get_cmap('RdBu_r').copy()
    colormap.set_over(color='lightgrey')
    colormap.set_under(color='lightgrey')
    #w, h = figaspect(1/1.3)
    #fig = plt.figure(figsize=(w, h), constrained_layout=True)

    x = pmf['d1'].to_numpy()
    y = pmf['d2'].to_numpy()
    z = pmf['energy'].to_numpy()

    binx = len(set(x))
    biny = len(set(y))
    x = x.reshape(binx, biny)
    y = y.reshape(binx, biny)
    energy = z.reshape(binx, biny)

    xi = energy_landscape['d1'].to_numpy()
    yi = energy_landscape['d2'].to_numpy()
    zi = energy_landscape['committor'].to_numpy()
   
    mask = (zi >= 0.49) & (zi <= 0.51)
    filtered_x = xi[mask]
    filtered_y = yi[mask]
    coefs = Polynomial.fit(filtered_x, filtered_y, deg=3)
    x_fit = np.linspace(min(filtered_x), max(filtered_x), 200)
    y_fit = coefs(x_fit)

    eps = 1.0e-16
    boundaries = np.linspace(0-eps, 1+eps, 21)
    norm = matplotlib.colors.BoundaryNorm(boundaries, ncolors=colormap.N, clip=True)

    cf = plt.scatter(xi, yi, c=zi, cmap=colormap, norm=norm, s=1.0)
    plt.contour(x, y, energy, colors='white', linewidths=1.0, alpha=0.8, levels=18)
    plt.plot(x_fit, y_fit, '--', color='darkgray', linewidth=1.5)

    ax = plt.gca()
    #ax.set_facecolor('lightgrey')
    ax.set_xlim(1.19, 3.5)
    ax.set_ylim(1.19, 3.5)
    ax.set_xlabel(r'$d1$ ($\AA$)',fontsize = 24)
    ax.set_ylabel(r'$d2$ ($\AA$)',fontsize = 24)

    ax.tick_params(direction='in', which='major', length=6.0, width=1.0, top=True, right=True)
    ax.tick_params(direction='in', which='minor', length=3.0, width=1.0, top=True, right=True)
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #ax.xaxis.set_major_locator(plt.FixedLocator(np.linspace(100, 160, 7)))
    #ax.yaxis.set_major_locator(plt.FixedLocator(np.linspace(10, 30, 3)))
    ax.set_aspect('equal', adjustable='box')
    clb = plt.colorbar(cf, ticks=np.linspace(0, 1, 6))
    clb.ax.set_title(r'q', pad=10.0,fontsize = 14)
    cf.set_clim(0, 1)
    plt.savefig(save_figure, dpi=300, bbox_inches='tight', transparent=False)
    #plt.show()
    return

def plot_pmf_hexbin(energy_landscape, pmf, save_figure=None):
    colormap = matplotlib.colormaps.get_cmap('RdBu_r').copy()
    colormap.set_over(color='lightgrey')
    colormap.set_under(color='lightgrey')

    x = pmf['d1'].to_numpy()
    y = pmf['d2'].to_numpy()
    z = pmf['energy'].to_numpy()

    binx = len(set(x))
    biny = len(set(y))
    x = x.reshape(binx, biny)
    y = y.reshape(binx, biny)
    energy = z.reshape(binx, biny)

    xi = energy_landscape['d1'].to_numpy()
    yi = energy_landscape['d2'].to_numpy()
    zi = energy_landscape['committor'].to_numpy()

    mask = (zi >= 0.49) & (zi <= 0.51)
    filtered_x = xi[mask]
    filtered_y = yi[mask]
    coefs = Polynomial.fit(filtered_x, filtered_y, deg=3)
    x_fit = np.linspace(min(filtered_x), max(filtered_x), 200)
    y_fit = coefs(x_fit)

    eps = 1.0e-16
    boundaries = np.linspace(0-eps, 1+eps, 21)
    norm = matplotlib.colors.BoundaryNorm(boundaries, ncolors=colormap.N, clip=True)

    fig, ax = plt.subplots()
    hb = ax.hexbin(xi, yi, C=zi, gridsize=100, cmap=colormap, reduce_C_function=np.mean, norm=norm)
    plt.contour(x, y, energy, colors='white', linewidths=1.0, alpha=0.8, levels=18)
    plt.plot(x_fit, y_fit, '--', color='darkgray', linewidth=1.5)

    ax.set_xlim(1.19, 3.5)
    ax.set_ylim(1.19, 3.5)

    ax.set_xlabel(r'$d1$ ($\AA$)', fontsize=24)
    ax.set_ylabel(r'$d2$ ($\AA$)', fontsize=24)
    ax.set_aspect('equal', adjustable='box')

    clb = plt.colorbar(hb, ticks=np.linspace(0, 1, 6))
    clb.ax.set_title(r'q', pad=10.0, fontsize=14)

    if save_figure:
        plt.savefig(save_figure, dpi=300, bbox_inches='tight', transparent=False)

plot_pmf_hexbin(data, pmf, save_figure=name)
#plot_pmf_smooth(data, save_figure='smooth.jpeg')
#plt.hist(data["committor"], bins=100, edgecolor='black')
#plt.xlabel("Committor")
#plt.ylabel("Frecuencia")
#plt.title("Histograma de Frecuencias")
#plt.savefig('hist.jpeg')
#plt.show()

