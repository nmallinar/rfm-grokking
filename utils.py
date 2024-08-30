import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt

def get_a_from_p(p):
    a_s = []
    vals = np.arange(1, p)
    for a in range(2, p):
        pows = (a**vals)%p
        pows = sorted(pows)
        # print(pows)
        if (pows==vals).all():
            a_s.append(a)
    return a_s

def get_lg_idx_from_p(p):
    a = get_a_from_p(p)[0]
    vals = (a**np.arange(1, p))%p
    lg_idx = [0]
    for n in np.arange(1, p):
        loc = np.where(vals == n)[0]+1
        lg_idx.append(loc.item())
    return np.array(lg_idx)

def reorder(M, p):
    lg_idx = get_lg_idx_from_p(p)
    E = np.zeros_like(M)
    for i in range(1,p):
        E[i,lg_idx[i]] = 1
    return E.T@M@E


def invert_idx(idx):
    # input: [1, 0, 2]
    # 0 -> 1
    # 2 -> 0
    # 1 -> 2
    #
    # output: [0, 1, 2]
    # 1 -> 0
    # 0 -> 2
    # 2 -> 1
    new_idx = np.zeros(len(idx))
    for i, j in enumerate(idx):
        new_idx[j] = i
    return new_idx.astype(np.int32)

def unorder(M, p):
    lg_idx = get_lg_idx_from_p(p)
    # print(lg_idx)
    lg_idx = invert_idx(lg_idx)
    # print(lg_idx)
    E = np.zeros_like(M)
    for i in range(1,p):
        E[i,lg_idx[i]] = 1
    return E.T@M@E

def gen_random_mult_circulant(p):
    C = np.zeros((p,p))
    row = np.random.uniform(size=(p-1,))
    row -= np.mean(row)
    circ = scipy.linalg.circulant(row)
    # print(circ.shape)
    C[1:,1:] = circ
    return unorder(C, p)

def gen_random_div_circulant(p):
    C = np.zeros((p,p))
    row = np.random.uniform(size=(p-1,))
    row -= np.mean(row)
    circ = scipy.linalg.circulant(row)
    # print(circ.shape)
    circ = torch.rot90(torch.from_numpy(circ)).numpy()

    C[1:,1:] = circ
    return unorder(C, p)

def display_all_agops(agops, per_class_agops, wandb, global_step, prefix=''):
    plt.clf()
    sqrt_agop = np.real(scipy.linalg.sqrtm(agops[0].numpy()))
    plt.imshow(sqrt_agop)
    plt.colorbar()
    img = wandb.Image(
        plt,
        caption='sqrt(AGOP)'
    )
    wandb.log({f'{prefix}sqrt_agop': img}, step=global_step)

    plt.clf()
    plt.imshow(sqrt_agop - np.diag(np.diag(sqrt_agop)))
    plt.colorbar()
    img = wandb.Image(
        plt,
        caption='no diag sqrt(AGOP)'
    )
    wandb.log({f'{prefix}no_diag_sqrt_agop': img}, step=global_step)

    for idx in range(len(per_class_agops)):
        sqrt_agop = np.real(scipy.linalg.sqrtm(per_class_agops[idx].numpy()))
        plt.clf()
        plt.imshow(sqrt_agop - np.diag(np.diag(sqrt_agop)))
        plt.colorbar()
        img = wandb.Image(
            plt,
            caption=f'cls_{idx}, sqrt(AGOP)'
        )
        wandb.log({
            f'per_class_agops/cls_{idx}_sqrt_agop': img
        }, step=global_step)

        vals, _ = np.linalg.eig(sqrt_agop)
        vals = np.array(sorted(list(vals))[::-1])
        plt.clf()
        #plt.plot(range(len(vals)), np.log(vals) + 1e-12)
        plt.plot(range(10), np.log(vals[:10]) + 1e-12)
        plt.xlabel('eig idx')
        plt.ylabel('ln(eigs)')
        plt.grid()
        img = wandb.Image(
            plt,
            caption=f'cls_{idx}, eigs sqrt(AGOP)'
        )
        wandb.log({
            f'per_class_agops_spectra/cls_{idx}_sqrt_agop': img
        }, step=global_step)
