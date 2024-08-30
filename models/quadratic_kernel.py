'''
Generalized quadratic formulation comes from the derivations given here:
https://arxiv.org/pdf/2209.04121
for the neural kernel with 2nd order Hermite activations with data not on the unit hypersphere.
See e.g. Theorem 1.
'''

import torch
from . import euclidean_distances, euclidean_distances_M
import numpy as np
import scipy
import time

torch.set_default_dtype(torch.float64)

def general_quadratic_M(samples, centers, M):
    samples_norm = (samples @ M)  * samples
    samples_norm = torch.sum(samples_norm, dim=1, keepdim=True)

    centers_norm = (centers @ M) * centers
    centers_norm = torch.sum(centers_norm, dim=1, keepdim=True)
    centers_norm = torch.reshape(centers_norm, (1, -1))

    return (1./2)*(samples_norm - 1)*(centers_norm - 1) + ((samples @ M) @ centers.T)**2

def general_quadratic_M_update(X, x, sol, P, batch_size=2,
                               centering=True, diag_only=False):
    x_norm = (x @ P) * x
    x_norm = torch.sum(x_norm, dim=1, keepdim=True)

    K = 2.0 * (X @ P) @ x.T

    a1 = sol.T
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape
    a1 = a1.reshape(n, c, 1)
    X1 = (X @ P).reshape(n, 1, d)
    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c*d)

    step2 = (x_norm - 1).T @ step1 + K.T @ step1
    del step1

    G = step2.reshape(-1, c, d)

    if centering:
        G_mean = torch.mean(G, axis=0).unsqueeze(0)
        G = G - G_mean
    M = 0.

    bs = batch_size
    batches = torch.split(G, bs)
    for i in range(len(batches)):
        if torch.cuda.is_available():
            grad = batches[i].cuda()
        else:
            grad = batches[i]

        gradT = torch.transpose(grad, 1, 2)
        if diag_only:
            T = torch.sum(gradT * gradT, axis=-1)
            M += torch.sum(T, axis=0).cpu()
        else:
            M += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    torch.cuda.empty_cache()
    M /= len(G)
    if diag_only:
        M = torch.diag(M)
    M = M.numpy()
    M = np.real(scipy.linalg.sqrtm(M))

    return torch.from_numpy(M)

def quadratic_M(samples, centers, M):
    return 3 * ((samples @ M) @ centers.T)**2

def quad_M_update(X, x, sol, P, batch_size=2,
                   centering=True, diag_only=False,
                   return_per_class_agop=False):
    M = 0.

    start = time.time()

    K = 3 * 2 * (X @ P @ x.T)**1
    a1 = sol.T
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape

    a1 = a1.reshape(n, c, 1)
    X1 = (X @ P).reshape(n, 1, d)
    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c*d)

    step2 = K.T @ step1
    del step1

    G = step2.reshape(-1, c, d)

    if centering:
        G_mean = torch.mean(G, axis=0).unsqueeze(0)
        G = G - G_mean
    M = 0.

    bs = batch_size
    batches = torch.split(G, bs)
    for i in range(len(batches)):
        if torch.cuda.is_available():
            grad = batches[i].cuda()
        else:
            grad = batches[i]

        gradT = torch.transpose(grad, 1, 2)
        if diag_only:
            T = torch.sum(gradT * gradT, axis=-1)
            M += torch.sum(T, axis=0).cpu()
        else:
            #gradT = torch.swapaxes(grad, 1, 2)#.cuda()
            M += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    torch.cuda.empty_cache()
    M /= len(G)

    per_class_agops = []
    if return_per_class_agop:
        for i in range(len(batches)):
            for class_i in range(G.shape[1]):
                if len(per_class_agops) < G.shape[1]:
                    per_class_agops.append(batches[i][:,class_i].T @ batches[i][:,class_i])
                else:
                    per_class_agops[class_i] += batches[i][:,class_i].T @ batches[i][:,class_i]
        for class_i in range(G.shape[1]):
            per_class_agops[class_i] /= len(G)
            per_class_agops[class_i] = torch.from_numpy(np.real(scipy.linalg.sqrtm(per_class_agops[class_i].numpy())))

    if diag_only:
        M = torch.diag(M)


    U, s, Vt = torch.linalg.svd(M)
    depth = 2
    alpha = (depth-1) / (depth)
    alpha = 0.5
    s = torch.pow(torch.abs(s), alpha)
    M = U @ torch.diag(s) @ Vt

    M = M.numpy()

    end = time.time()

    return torch.from_numpy(M), per_class_agops
