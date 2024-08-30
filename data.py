# based on:
# https://github.com/danielmamay/grokking/blob/main/grokking/data.py

from math import ceil
import torch
import itertools
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.set_default_dtype(torch.float64)

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x),
    "(x//y)if(y%2==1)else(x-y)": lambda x, y, _: torch.where(y % 2 == 1, x // y, x - y)
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    "x*y": lambda x, y, _: (x, y, x*y),
    **DIVISION_MODULO_OPERATIONS,
    "x^2+y": lambda x, y, _: (x, y, x**2 + y),
    "x^2+y^2": lambda x, y, _: (x, y, x**2 + y**2),
    "x^2+xy+y^2": lambda x, y, _: (x, y, x**2 + x*y + y**2),
    "x^2+xy+y^2+x": lambda x, y, _: (x, y, x**2 + x*y + y**2 + x),
    "x^3+xy": lambda x, y, _: (x, y, x**3 + x*y),
    "x^3+xy^2+x": lambda x, y, _: (x, y, x**3 + x*y**2 + y)
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
}

def operation_mod_p_data(operation: str, p: int):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x = torch.arange(0, p)
    y = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)
    x, y = torch.cartesian_prod(x, y).T

    x, y, z = ALL_OPERATIONS[operation](x, y, p)
    results = z.remainder(p)

    inputs = torch.stack([x, y], dim=1)
    labels = results

    return inputs, labels

def multitask_op_mod_p_data(op1, op2, p, train_frac_per_op):
    inp1, lab1 = operation_mod_p_data(op1, p)
    X_tr1, y_tr1, X_te1, y_te1 = make_data_splits(inp1, lab1, train_frac_per_op)
    X_tr1 = F.one_hot(X_tr1, p).view(-1, 2*p).double()
    X_te1 = F.one_hot(X_te1, p).view(-1, 2*p).double()
    X_tr2 = X_tr1.clone()
    X_te2 = X_te1.clone()

    task2_dim = p

    zeros = torch.zeros((X_tr1.shape[0],1))
    X_tr1 = torch.hstack((X_tr1, zeros))
    y_tr1 = F.one_hot(y_tr1, p).double()

    zeros = torch.zeros((X_te1.shape[0],1))
    X_te1 = torch.hstack((X_te1, zeros))
    y_te1 = F.one_hot(y_te1, p).double()

    ones = torch.ones((X_tr2.shape[0], 1))
    dig1 = X_tr2[:,:p].argmax(-1)
    dig2 = X_tr2[:,p:].argmax(-1)

    _, _, y_tr2 = ALL_OPERATIONS[op2](dig1, dig2, p)
    y_tr2 = y_tr2.remainder(p)

    y_tr2 = F.one_hot(y_tr2, task2_dim).double()
    X_tr2 = torch.hstack((X_tr2, ones))

    ones = torch.ones((X_te2.shape[0], 1))
    dig1 = X_te2[:,:p].argmax(-1)
    dig2 = X_te2[:,p:].argmax(-1)
    _, _, y_te2 = ALL_OPERATIONS[op2](dig1, dig2, p)
    y_te2 = y_te2.remainder(p)

    y_te2 = F.one_hot(y_te2, task2_dim).double()
    X_te2 = torch.hstack((X_te2, ones))

    X_tr = torch.vstack((X_tr1, X_tr2))
    y_tr = torch.vstack((y_tr1, y_tr2))

    return X_tr, y_tr, X_te1, y_te1, X_te2, y_te2

def make_data_splits(inputs, labels, training_fraction):
    train_size = int(training_fraction * inputs.shape[0])
    val_size = inputs.shape[0] - train_size

    perm = torch.randperm(inputs.shape[0])
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    return inputs[train_idx], labels[train_idx], inputs[val_idx], labels[val_idx]

def make_dataloader(inputs, labels, batch_size, shuffle=False, drop_last=False):
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    batch_size = min(batch_size, ceil(len(dataset) / 2))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
