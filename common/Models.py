import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree
from gvp import GVP, GVPConvLayer
from common.BaseGVP import GVPModel

class Model(nn.Module):
    def __init__(self, hidden_dim=128, hidden_dim1=64, node_s_dim=1, node_v_dim=1, edge_s_dim=1, edge_v_dim=1, 
                 linear_layers=2, output_dim=1, conv_layers=2, dropout=0.2):
        super().__init__()

        self.node_encoder = GVP((node_s_dim, node_v_dim), (hidden_dim, hidden_dim))
        self.edge_encoder = GVP((edge_s_dim, edge_v_dim), (hidden_dim, hidden_dim))

        self.convs = nn.ModuleList([
            GVPConvLayer((hidden_dim, hidden_dim), (hidden_dim, hidden_dim)) 
            for _ in range(conv_layers)
        ])
        self.pool = global_mean_pool

        mlp_layers = []
        mlp_layers.append(nn.Linear(hidden_dim, hidden_dim1))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(dropout))

        for _ in range(linear_layers - 1):  
            mlp_layers.append(nn.Linear(hidden_dim1, hidden_dim1))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))

        mlp_layers.append(nn.Linear(hidden_dim1, output_dim))  
        mlp_layers.append(nn.Sigmoid())  

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, node_s, node_v, edge_index, edge_s, edge_v, batch):
        if node_v.dim() == 2:
            node_v = node_v.unsqueeze(1)
        if edge_v.dim() == 2:
            edge_v = edge_v.unsqueeze(1)
            
        nodes = self.node_encoder((node_s, node_v))
        edges = self.edge_encoder((edge_s, edge_v))

        for conv in self.convs:
            x = conv(nodes, edge_index, edges)

        x_scalar, _ = x
        x_scalar = self.pool(x_scalar, batch)

        out = self.mlp(x_scalar)
        return out

class VCN(nn.Module):
    def __init__(
        self,
        inputs: int = 6,
        outputs: int = 1,
        hidden_dim: int = 32,
        n_layers: int = 6,
        drop_rate: float = 0.1,
        activation: str = 'ELU',
    ) -> None:
        super().__init__()
        
        activations = {
            'ELU': nn.ELU(),
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid(),
        }
        
        if activation not in activations:
            raise ValueError("Invalid activation}")

        self.activation = activations[activation]
        self.sigmoid = nn.Sigmoid()

        layers = [nn.Linear(inputs, hidden_dim), self.activation]
        for _ in range(n_layers - 1):  
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
        
        layers.append(nn.Linear(hidden_dim, outputs))
        layers.append(self.sigmoid)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    @torch.jit.export
    def q_layer(self, x):
        return self.sigmoid(self.model(x))

            
class GNNModel(GVPModel):
    def __init__(
        self,
        n_cvs: int = 1,
        gnn_out: int = 1,
        node_s_dim: int = 1,
        node_v_dim: int = 1,
        edge_s_dim: int = 1,
        edge_v_dim: int = 1,
        n_layers: int = 2,
        n_messages: int = 3,
        n_feedforwards: int = 2,
        drop_rate: float = 0.1,
        activation: str = 'ReLU',
    ) -> None:
        super().__init__(
            n_out=gnn_out,
            node_s_dim=node_s_dim,
            node_v_dim=node_v_dim,
            edge_s_dim=edge_s_dim,
            edge_v_dim=edge_v_dim,
            n_layers=n_layers, 
            n_messages=n_messages, 
            n_feedforwards=n_feedforwards, 
            drop_rate=drop_rate, 
            activation=activation
        )

        #self.ffn = VCN(inputs=gnn_out, outputs=n_cvs)

    def forward(self, node_s, node_v, edge_index, edge_s, edge_v, batch):
        y = self.forward_gnn(node_s, node_v, edge_index, edge_s, edge_v, batch)
        #y = self.ffn(y)
        return y

    @torch.jit.export
    def q(self, node_s, node_v, edge_index, edge_s, edge_v, batch):
        y = self.forward_gnn(node_s, node_v, edge_index, edge_s, edge_v, batch)
        #y = self.ffn.q_layer(y)
        return y


