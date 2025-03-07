import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree
from gvp import GVP, GVPConvLayer, LayerNorm

class GVPModel(torch.nn.Module):
    def __init__(
        self,
        n_out: int = 1,
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
        super(GVPModel, self).__init__()

        ACTIVATIONS = {
            "ReLU": nn.ReLU(),
            "ELU": nn.ELU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
        }
        
        activation_fn = ACTIVATIONS.get(activation, nn.ReLU())
        
        self.node_embedding = nn.ModuleList([
            LayerNorm((1, 1)),
            GVP(
                (1, 1),
                (node_s_dim, node_v_dim),
                activations=(None, None),
                vector_gate=True,
            )
        ])

        self.edge_embedding = nn.ModuleList([
            LayerNorm((1, 1)),
            GVP(
                (1, 1),
                (edge_s_dim, edge_v_dim),
                activations=(None, None),
                vector_gate=True,
            )
        ])
        
        self.gvp_layers = nn.ModuleList([
            GVPConvLayer(
                (node_s_dim, node_v_dim), 
                (edge_s_dim, edge_v_dim),
                n_message=n_messages,
                n_feedforward=n_feedforwards,
                drop_rate=drop_rate,
                activations=(activation_fn, None),
                vector_gate=True,
            ) for _ in range(n_layers)
        ])


        self.out_layer = GVP(
                (node_s_dim, node_v_dim),
                (n_out, 0),
                activations=(None, None),
                vector_gate=True,
            )

        self.pool = global_mean_pool

    def forward_gnn(self, node_s, node_v, edge_index, edge_s, edge_v, batch):
        if node_v.dim() == 2:
            node_v = node_v.unsqueeze(1)
        if edge_v.dim() == 2:
            edge_v = edge_v.unsqueeze(1)

        for layer in self.node_embedding:
            node_s, node_v = layer((node_s, node_v))
            
        for layer in self.edge_embedding:
            edge_s, edge_v = layer((edge_s, edge_v))

        for layer in self.gvp_layers:
            node_s, node_v = layer((node_s, node_v), edge_index, (edge_s, edge_v))
        
        output = self.out_layer((node_s, node_v))
        
        if batch is None:
            batch = torch.zeros(node_s.shape[0], dtype=torch.long, device=node_s.device)
        
        output = self.pool(output, batch)
        return output.squeeze()

    def forward(self, node_s, node_v, edge_index, edge_s, edge_v, batch):
        return self.forward_gnn(node_s, node_v, edge_index, edge_s, edge_v, batch)

