import torch
from . import euclidean_distances, euclidean_distances_M
import numpy as np
import scipy

torch.set_default_dtype(torch.float64)

def laplacian(samples, centers, bandwidth, return_dist=False):
    '''Laplacian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=False)

    if return_dist:
        dist = kernel_mat.clone()

    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()

    if return_dist:
        return kernel_mat, dist

    return kernel_mat

def laplacian_M(samples, centers, bandwidth, M, return_dist=False):
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)

    if return_dist:
        dist = kernel_mat.clone()

    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()

    if return_dist:
        return kernel_mat, dist

    return kernel_mat

def get_grads(X, sol, L, P, batch_size=2, K=None, dist=None, centering=False):
    M = 0.

    x = X

    if K is None:
        K = laplacian_M(X, x, L, P)

    if dist is None:
        dist = euclidean_distances_M(X, x, P, squared=False)

    dist = torch.where(dist < 1e-10, torch.zeros(1).float(), dist)
    K = K/dist
    K[K == float("Inf")] = 0.
    K[torch.isnan(K)] = 0.

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

    step2 = step2.reshape(-1, c, d)

    a2 = sol
    step3 = (a2 @ K).T

    del K, a2

    step3 = step3.reshape(m, c, 1)
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * -1/L

    M = 0.

    if centering:
        G = G - G.mean(0)

    bs = batch_size
    batches = torch.split(G, bs)
    for i in range(len(batches)):
        # grad = batches[i].cuda()
        grad = batches[i]
        gradT = torch.transpose(grad, 1, 2)
        M += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    torch.cuda.empty_cache()
    M /= len(G)
    M = np.real(scipy.linalg.sqrtm(M.numpy()))

    return torch.from_numpy(M)

def laplacian_M_update(samples, centers, bandwidth, M, weights, K=None, dist=None, \
                       centers_bsize=-1, centering=False):
    return get_grads(samples, weights.T, bandwidth, M, K=K, centering=centering)
