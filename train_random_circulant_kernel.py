import os
import sys
import argparse
import wandb
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import random

from data import operation_mod_p_data, make_data_splits
from models import laplace_kernel, gaussian_kernel, jax_ntk_nngp, quadratic_kernel, euclidean_distances_M
import utils

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
# torch.manual_seed(3143)
# random.seed(253)
# np.random.seed(1145)

def eval(sol, K, y_onehot):
    preds = np.real(K.T @ sol)
    loss = (preds - y_onehot).pow(2).mean()

    corr = 0
    if y_onehot.shape[1] > 1:
        count = torch.sum(y_onehot.argmax(-1) == preds.argmax(-1))
        acc = count / y_onehot.shape[0]
    else:
        acc = 0.0

    return acc, loss, corr

def get_test_kernel(X_tr, X_te, M, bandwidth, ntk_depth, kernel_type):
    K_test = None
    if kernel_type == 'laplace':
        K_test = laplace_kernel.laplacian_M(X_tr, X_te, bandwidth, M, return_dist=False)
    elif kernel_type == 'gaussian':
        K_test = gaussian_kernel.gaussian_M(X_tr, X_te, bandwidth, M)
    elif kernel_type == 'jax_ntk_nngp':
        _, K_test = jax_ntk_nngp.ntk_fn(X_tr, X_te, M=M, depth=ntk_depth, bias=0, convert=True)
    elif kernel_type == 'quadratic':
        K_test = quadratic_kernel.quadratic_M(X_tr, X_te, M)
    elif kernel_type == 'general_quadratic':
        K_test = quadratic_kernel.general_quadratic_M(X_tr, X_te, M)

    return K_test

def solve(X_tr, y_tr_onehot, M, bandwidth, ntk_depth, kernel_type,
          ridge=1e-3):

    K_train = None
    dist = None
    sol = None

    if kernel_type == 'laplace':
        K_train, dist = laplace_kernel.laplacian_M(X_tr, X_tr, bandwidth, M, return_dist=True)
    elif kernel_type == 'gaussian':
        K_train = gaussian_kernel.gaussian_M(X_tr, X_tr, bandwidth, M)
    elif kernel_type == 'jax_ntk_nngp':
        _, K_train = jax_ntk_nngp.ntk_fn(X_tr, X_tr, M=M, depth=ntk_depth, bias=0, convert=True)
    elif kernel_type == 'quadratic':
        K_train = quadratic_kernel.quadratic_M(X_tr, X_tr, M)
    elif kernel_type == 'general_quadratic':
        K_train = quadratic_kernel.general_quadratic_M(X_tr, X_tr, M)

    sol = torch.from_numpy(np.linalg.solve(K_train.numpy() + ridge * np.eye(len(K_train)), y_tr_onehot.numpy()).T)
    sol = sol.T
    preds = K_train @ sol

    return sol, K_train, dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', default='WANDB_ORG_NAME')
    parser.add_argument('--wandb_proj_name', default='rfm_grokking')
    parser.add_argument('--group_key', default='')
    parser.add_argument('--out_dir', default='./wandb')
    parser.add_argument('--operation', '-op', default="x+y")
    parser.add_argument('--prime', '-p', default=61, type=int)
    parser.add_argument('--training_fraction', default=0.5, type=float)
    parser.add_argument('--ridge', default=0.0, type=float)
    parser.add_argument('--bandwidth', default=2.5, type=float)
    parser.add_argument('--ntk_depth', default=2, type=int)
    parser.add_argument('--kernel_type', default='gaussian', choices={'gaussian', 'laplace', 'jax_ntk_nngp', 'quadratic', 'general_quadratic'})
    parser.add_argument('--wandb_offline', default=False, action='store_true')
    args = parser.parse_args()

    mode = 'online'
    if args.wandb_offline:
        mode = 'offline'

    wandb.init(entity=args.wandb_entity, project=args.wandb_proj_name, mode=mode, config=args,
               dir=args.out_dir)

    wandb.run.name = f'{wandb.run.id} - training frac: {args.training_fraction}, p: {args.prime}'

    all_inputs, all_labels = operation_mod_p_data(args.operation, args.prime)
    X_tr, y_tr, X_te, y_te = make_data_splits(all_inputs, all_labels, args.training_fraction)

    p = args.prime
    X_tr = F.one_hot(X_tr, args.prime).view(-1, 2*args.prime).double()
    y_tr_onehot = F.one_hot(y_tr, args.prime).double()
    X_te = F.one_hot(X_te, args.prime).view(-1, 2*args.prime).double()
    y_te_onehot = F.one_hot(y_te, args.prime).double()

    M = torch.zeros(2*args.prime, 2*args.prime).double()

    if args.operation == 'x-y':
        col = torch.rand(args.prime)
        col -= col.mean()
        circ = torch.from_numpy(scipy.linalg.circulant(col.numpy()))
        M[:p,p:] = torch.rot90(circ)
    elif args.operation == 'x+y':
        col = torch.rand(args.prime)
        col -= col.mean()
        circ = torch.from_numpy(scipy.linalg.circulant(col.numpy()))
        M[:p,p:] = circ
    elif args.operation == 'x*y':
        circ = utils.gen_random_mult_circulant(p)
        circ = torch.from_numpy(circ)
        M[:p,p:] = circ
    elif args.operation == 'x/y':
        circ = utils.gen_random_div_circulant(p)
        circ = torch.from_numpy(circ)
        M[:p,p:] = circ
    M[p:,:p] = M[:p,p:].clone().T

    M[:p, :p] = torch.eye(p) - 1./p * torch.ones(p, p)
    M[p:, p:] = M[:p, :p]

    M = torch.from_numpy(np.real(scipy.linalg.sqrtm(M)))

    sol, K_train, dist = solve(X_tr, y_tr_onehot, M, args.bandwidth, args.ntk_depth, args.kernel_type,
                               ridge=args.ridge)

    acc, loss, corr = eval(sol, K_train, y_tr_onehot)
    print(f'Train MSE:\t{loss}')
    print(f'Train Acc:\t{acc}')

    wandb.log({
        'training/accuracy': acc,
        'training/loss': loss
    })

    K_test = get_test_kernel(X_tr, X_te, M, args.bandwidth, args.ntk_depth, args.kernel_type)

    acc, loss, corr = eval(sol, K_test, y_te_onehot)
    print(f'Test MSE:\t{loss}')
    print(f'Test Acc:\t{acc}')
    print()

    wandb.log({
        'validation/accuracy': acc,
        'validation/loss': loss
    })

if __name__=='__main__':
    main()
