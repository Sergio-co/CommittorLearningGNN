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

def JAB1(q_0, q_t, w0, wt):
    L = torch.mean(torch.square(torch.sqrt(w0)*q_0 - torch.sqrt(wt)*q_t))
    return L

def loss(model, data_0, data_t, labels, kforce=1.0):
    k_a0, k_b0, center_0, weight_0, k_at, k_bt, center_t, weight_t, weights = labels

    q_0 = model(data_0)
    q_t = model(data_t)

    #weights = torch.clamp(weights, min=1e-6)
    #weights = torch.ones_like(q_0, device=q_0.device)

    loss = JAB(q_0, q_t, weights)
    #loss = JAB(q_0, q_t, weight_0, weight_t)
    # add harmonic restraint for basin A
    res_A = kforce*k_a0 * torch.square(q_0 - center_0) + kforce*k_at * torch.square(q_t - center_t)
    # add harmonic restraint for basin B
    res_B = kforce*k_b0 * torch.square(q_0 - center_0) + kforce*k_bt * torch.square(q_t - center_t)
    # weight the restraints
    res = torch.sum(weights * (res_A + res_B)) / torch.sum(weights)
    spread = torch.var(q_0)

    total_loss = loss + res #+ spread

    return total_loss, q_0[0], loss, res, spread

