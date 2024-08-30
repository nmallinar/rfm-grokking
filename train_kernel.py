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
from models import laplace_kernel, gaussian_kernel, \
                   jax_ntk_nngp, quadratic_kernel
import utils

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
# torch.manual_seed(3143)
# random.seed(253)
# np.random.seed(1145)

def eval(sol, K, y_onehot):
    preds = K.T @ sol
    loss = (preds - y_onehot).pow(2).mean()

    labels = y_onehot.argmax(-1)
    correct_logit_loss = 0.0
    for idx in range(len(labels)):
        correct_logit_loss += (preds[idx][labels[idx]] - 1).pow(2)
    correct_logit_loss /= labels.shape[0]

    corr = 0
    if y_onehot.shape[1] > 1:
        count = torch.sum(y_onehot.argmax(-1) == preds.argmax(-1))
        acc = count / y_onehot.shape[0]
    elif y_onehot.shape[1] == 1 or len(y_onehot.shape) == 1:
        count = torch.sum((y_onehot > 0.5) == (preds > 0.5))
        acc = count / y_onehot.shape[0]
    else:
        acc = 0.0

    return acc, loss, corr, correct_logit_loss

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

def update(samples, centers, bandwidth, M, weights, K, dist, \
           kernel_type, ntk_depth, centers_bsize=-1, centering=False,
           agop_power=0.5):
    if kernel_type == 'laplace':
        M = laplace_kernel.laplacian_M_update(samples, centers, bandwidth, M, weights, K=K, dist=dist, \
                                   centers_bsize=centers_bsize, centering=centering)
        per_class_agops = []
    elif kernel_type == 'gaussian':
        M, per_class_agops = gaussian_kernel.gaussian_M_update(samples, centers, bandwidth, M, weights, K=K, \
                              centers_bsize=centers_bsize, centering=centering, agop_power=agop_power,
                              return_per_class_agop=False)
    elif kernel_type == 'jax_ntk_nngp':
        M = jax_ntk_nngp.ntk_relu_M_update(weights, centers, samples, M, ntk_depth=ntk_depth)
        per_class_agops = []
    elif kernel_type == 'quadratic':
        M, per_class_agops = quadratic_kernel.quad_M_update(samples, centers, weights.T, M, centering=centering,
                                                            return_per_class_agop=False)
    elif kernel_type == 'general_quadratic':
        M = quadratic_kernel.general_quadratic_M_update(samples, centers, weights.T, M, centering=centering)
        per_class_agops = []
    return M, per_class_agops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_entity', default='WANDB_ORG_NAME')
    parser.add_argument('--wandb_proj_name', default='rfm_grokking')
    parser.add_argument('--wandb_offline', default=False, action='store_true')
    parser.add_argument('--group_key', default='', type=str)
    parser.add_argument('--out_dir', default='./wandb')
    parser.add_argument('--operation', '-op', default="x+y")
    parser.add_argument('--prime', '-p', default=61, type=int)
    parser.add_argument('--training_fraction', default=0.5, type=float)
    parser.add_argument('--iters', default=50, type=int)
    parser.add_argument('--ridge', default=0.0, type=float)
    parser.add_argument('--bandwidth', default=2.5, type=float)
    parser.add_argument('--ntk_depth', default=2, type=int)
    parser.add_argument('--kernel_type', default='gaussian', choices={'gaussian', 'laplace', 'quadratic',
                                                                      'general_quadratic', 'jax_ntk_nngp'})
    parser.add_argument('--save_agops', default=False, action='store_true')
    parser.add_argument('--agop_power', default=0.5, type=float)
    args = parser.parse_args()

    mode = 'online'
    if args.wandb_offline:
        mode = 'offline'

    wandb.init(entity=args.wandb_entity, project=args.wandb_proj_name, mode=mode, config=args,
               dir=args.out_dir)

    out_dir = os.path.join(args.out_dir, args.wandb_proj_name, wandb.run.id)
    os.makedirs(out_dir, exist_ok=True)

    wandb.run.name = f'{wandb.run.id} - p: {args.prime}, train_frac: {args.training_fraction}, ' + \
                     f'agop_power: {args.agop_power}, ridge: {args.ridge}, bdwth: {args.bandwidth}'

    all_inputs, all_labels = operation_mod_p_data(args.operation, args.prime)
    X_tr, y_tr, X_te, y_te = make_data_splits(all_inputs, all_labels, args.training_fraction)

    X_tr = F.one_hot(X_tr, args.prime).view(-1, 2*args.prime).double()
    y_tr_onehot = F.one_hot(y_tr, args.prime).double()

    X_te = F.one_hot(X_te, args.prime).view(-1, 2*args.prime).double()
    y_te_onehot = F.one_hot(y_te, args.prime).double()

    M = torch.eye(X_tr.shape[1]).double()

    for rfm_iter in range(args.iters):
        sol, K_train, dist = solve(X_tr, y_tr_onehot, M, args.bandwidth, args.ntk_depth, args.kernel_type,
                                   ridge=args.ridge)

        acc, loss, corr, correct_logit_loss = eval(sol, K_train, y_tr_onehot)
        print(f'Round {rfm_iter} Train MSE:\t{loss}')
        print(f'Round {rfm_iter} Train Acc:\t{acc}')
        wandb.log({
            'training/accuracy': acc,
            'training/loss': loss,
            'training/correct_logit_loss': correct_logit_loss,
        }, step=rfm_iter)

        K_test = get_test_kernel(X_tr, X_te, M, args.bandwidth, args.ntk_depth, args.kernel_type)

        acc, loss, corr, correct_logit_loss = eval(sol, K_test, y_te_onehot)
        print(f'Round {rfm_iter} Test MSE:\t{loss}')
        print(f'Round {rfm_iter} Test Acc:\t{acc}')
        print()

        wandb.log({
            'validation/accuracy': acc,
            'validation/loss': loss,
            'validation/correct_logit_loss': correct_logit_loss,
        }, step=rfm_iter)

        M, per_class_agops = update(X_tr, X_tr, args.bandwidth, M, sol, K_train, dist, \
                       args.kernel_type, args.ntk_depth, centers_bsize=-1, centering=True,
                       agop_power=args.agop_power)

        with torch.no_grad():
            wandb.log({
                'training/agop_tr': torch.trace(M)
            }, step=rfm_iter)

        # if (rfm_iter < 31) or \
        if (rfm_iter < 100 and rfm_iter % 25 == 0) or \
            (rfm_iter < 500 and rfm_iter % 50 == 0):

            if args.save_agops:
                os.makedirs(os.path.join(out_dir, f'iter_{rfm_iter}'), exist_ok=True)
                np.save(os.path.join(out_dir, f'iter_{rfm_iter}/M.npy'), M.numpy())

                if len(per_class_agops) > 0:
                    subdir = os.path.join(out_dir, f'iter_{rfm_iter}', 'per_class_agops')
                    os.makedirs(subdir, exist_ok=True)
                    for cls_idx in range(len(per_class_agops)):
                        np.save(os.path.join(subdir, f'M_cls_{cls_idx}.npy'), per_class_agops[cls_idx].numpy())

            if not args.wandb_offline:
                plt.clf()
                plt.imshow(M)
                plt.colorbar()
                img = wandb.Image(
                    plt,
                    caption=f'M'
                )
                wandb.log({'M': img}, step=rfm_iter)

                plt.clf()
                plt.imshow(M - torch.diag(torch.diag(M)))
                plt.colorbar()
                img = wandb.Image(
                    plt,
                    caption=f'M_no_diag'
                )
                wandb.log({'M_no_diag': img}, step=rfm_iter)

                # for cls_idx in range(len(per_class_agops)):
                #     plt.clf()
                #     plt.imshow(per_class_agops[cls_idx] - torch.diag(torch.diag(per_class_agops[cls_idx])))
                #     plt.colorbar()
                #     img = wandb.Image(
                #         plt,
                #         caption=f'cls_{cls_idx} M_no_diag'
                #     )
                #     wandb.log({f'per_class/{cls_idx}_M_no_diag': img}, step=rfm_iter)

                # M_vals = torch.flip(torch.linalg.eigvalsh(M), (0,))
                #
                # plt.clf()
                # plt.plot(range(len(M_vals)), np.log(M_vals))
                # plt.grid()
                # plt.xlabel('eigenvalue idx')
                # plt.ylabel('ln(eigenvalue)')
                # img = wandb.Image(
                #     plt,
                #     caption='M_eigenvalues'
                # )
                # wandb.log({'M_eigs': img}, step=rfm_iter)

if __name__=='__main__':
    main()
