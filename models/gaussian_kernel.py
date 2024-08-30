import torch
from . import euclidean_distances, euclidean_distances_M
import numpy as np
import scipy

torch.set_default_dtype(torch.float64)

def gaussian(samples, centers, bandwidth, return_dist=False):
    '''Gaussian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers)

    if return_dist:
        dist = kernel_mat.clone()

    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()

    if return_dist:
        return kernel_mat, dist

    return kernel_mat

def gaussian_M(samples, centers, bandwidth, M, return_dist=False):
    # assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=True)

    if return_dist:
        dist = kernel_mat.clone()

    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()

    if return_dist:
        return kernel_mat, dist

    return kernel_mat

def get_grads(X, sol, L, P, batch_size=2, K=None, centering=False, x=None,
              agop_power=0.5, return_per_class_agop=False):
    M = 0.

    if x is None:
        x = X

    if K is None:
        K = gaussian_M(X, x, L, P)


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

    G = (step2 - step3) * -1/(L**2)

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

    if agop_power == 0.5:
        M = np.real(scipy.linalg.sqrtm(M.numpy()))
    elif agop_power.is_integer():
        if agop_power == 1:
            M = M.numpy()
        else:
            M = np.real(np.linalg.matrix_power(M.numpy(), int(agop_power)))
    else:
        eigs, vecs = np.linalg.eigh(M.numpy())
        eigs = np.power(eigs, agop_power)
        eigs[np.isnan(eigs)] = 0.0
        M = vecs @ np.diag(eigs) @ vecs.T

    return torch.from_numpy(M), per_class_agops

def gaussian_M_update(samples, centers, bandwidth, M, weights, K=None, \
                      centers_bsize=-1, centering=False, agop_power=0.5,
                      return_per_class_agop=False):
    return get_grads(samples, weights.T, bandwidth, M, K=K, centering=centering, x=centers,
                     agop_power=agop_power, return_per_class_agop=return_per_class_agop)
