from neural_tangents import stax
import jax.dlpack
from jax import jacfwd, jacrev
from jax.scipy.linalg import sqrtm
import jax.numpy as jnp
import torch
import numpy as np
import scipy
import math
import warnings
warnings.filterwarnings("ignore")

def torch2jax(x):
    x = torch.to_dlpack(x)
    return jax.dlpack.from_dlpack(x)

def jax2torch(x):
    x = jax.dlpack.to_dlpack(x)
    return torch.utils.dlpack.from_dlpack(x)

'''
Right now just manually defining depth and activation...
'''
layers = []
for _ in range(1, 2):
    layers += [
        stax.Dense(1, W_std=1, b_std=None, parameterization='ntk'),
        # stax.ElementwiseNumerical(lambda x: (jnp.pow(x, 2) - 1.0)/jnp.sqrt(2.0), 100)
        # stax.Relu()
        stax.Monomial(2)
        # stax.Hermite(2)
        # stax.Erf()
        # stax.Rbf(2.5)
        # stax.Cos()
        # stax.LeakyRelu(0.1)
        # stax.Gaussian()
    ]
layers.append(stax.Dense(1, W_std=1, b_std=None, parameterization='ntk'))

_, _, kernel_fn = stax.serial(*layers)

def ntk_fn(x, y, M=None, depth=2, bias=1, convert=True):
    if convert:
        x = torch2jax(x)
        y = torch2jax(y)
        M = torch2jax(M)

    if M is not None:
        # sqrtM = M
        sqrtM = jnp.real(sqrtm(M))
        x = x @ sqrtM
        y = y @ sqrtM
        # x /= jnp.linalg.norm(x, axis=1, keepdims=True)
        # y /= jnp.linalg.norm(y, axis=1, keepdims=True)

    ntk_nngp_jax = kernel_fn(x, y)
    nngp_jax = ntk_nngp_jax.nngp
    ntk_jax = ntk_nngp_jax.ntk

    if convert:
        nngp_jax = jax2torch(nngp_jax)
        ntk_jax = jax2torch(ntk_jax)

        return nngp_jax.double(), ntk_jax.double()

    return nngp_jax, ntk_jax

def ntk_relu_M_update(alphas, centers, samples, M, ntk_depth=1):

    # centers = torch2jax(centers)
    # samples = torch2jax(samples)
    M = torch2jax(M)

    M_is_passed = M is not None
    # M_is_passed = False
    sqrtM = None
    if M_is_passed:
        # sqrtM = utils.matrix_sqrt(M)
        sqrtM = jnp.real(sqrtm(M))
        # sqrtM = M

    def get_solo_grads(sol, X, x):
        if M_is_passed:
            X_M = X@sqrtM
            # X_M /= jnp.linalg.norm(X_M, axis=1, keepdims=True)
        else:
            X_M = X

        def egop_fn(z):
            if M_is_passed:
                z_ = z@sqrtM
                # z_ /= jnp.linalg.norm(z_, axis=1, keepdims=True)
            else:
                z_ = z
            _, K = ntk_fn(z_, X_M, M=M, depth=ntk_depth, bias=1, convert=False)
            return (K@sol).squeeze()
        # grads = jax.jacrev(egop_fn)(x)
        grads = jnp.squeeze(jax.vmap(jax.jacrev(egop_fn))(jnp.expand_dims(x, 1)))
        grads = jnp.nan_to_num(grads)
        # grads = torch.vmap(torch.func.jacrev(egop_fn))(x.unsqueeze(1)).squeeze()
        # grads = torch.nan_to_num(grads)
        return jax2torch(grads)

    n, d = centers.shape
    s = len(samples)

    chunk = 1000
    train_batches = torch.split(torch.arange(n), chunk)

    egop = 0
    egip = 0
    G = 0
    for btrain in train_batches:
        G += get_solo_grads(torch2jax(alphas[btrain,:]), torch2jax(centers[btrain]), torch2jax(samples))
    c = G.shape[1]
    G = G - G.mean(0).unsqueeze(0)

    Gd = G.reshape(-1, d)
    egop += Gd.T @ Gd/s

    egop = np.real(scipy.linalg.sqrtm(egop.numpy()))

    # return egop
    return torch.from_numpy(egop)
