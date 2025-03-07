#!/usr/bin/env python3
import torch
import numpy as np


#@torch.compile
def renormalize_weights(weights):
    import numpy as np
    return weights / np.sum(weights) * np.shape(weights)[0]

def JAB(q_0, q_t, weights):
    L = torch.sum(weights * torch.square(q_0 - q_t))
    return L / torch.sum(weights)

def loss(model, data_0, data_t, labels, kforce=1.0):
    k_a0, k_b0, center_0, weight_0, k_at, k_bt, center_t, weight_t = labels
    weights = torch.sqrt(weight_0 * weight_t)

    q_0 = model(data_0.node_s, data_0.node_v, data_0.edge_index, data_0.edge_attr, data_0.edge_v, data_0.batch)
    q_t = model(data_t.node_s, data_t.node_v, data_t.edge_index, data_t.edge_attr, data_t.edge_v, data_t.batch)

    loss = JAB(q_0, q_t, weights)
    # add harmonic restraint for basin A
    res_A = kforce*k_a0 * torch.square(q_0 - center_0) + kforce*k_at * torch.square(q_t - center_t)
    # add harmonic restraint for basin B
    res_B = kforce*k_b0 * torch.square(q_0 - center_0) + kforce*k_bt * torch.square(q_t - center_t)
    # weight the restraints
    res = torch.sum(weights * (res_A + res_B)) / torch.sum(weights)
    
    return loss+res, torch.min(q_0), torch.max(q_0)

