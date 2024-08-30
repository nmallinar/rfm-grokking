# Grokking modular arithmetic with recursive feature machines (RFM)

Code for the the paper "Emergence in non-neural models: grokking modular arithmetic via average gradient outer product" which can be found [here](https://arxiv.org/abs/2407.20199).

This repository was built off of [this](https://github.com/danielmamay/grokking) base repository.

## Installation details

Packaged can be found in `requirements.txt` and installed via: `pip install -r requirements.txt`.

### neural-tangents

See [neural-tangents](https://github.com/google/neural-tangents) for more details on installation of Jax for GPU vs. CPU in case of problems.

## Running

Use `--wandb_offline` to disable logging to [wandb](https://wandb.ai).



## To do (last updated: 8/30/2024)

- There is a bug in the `update` function of `jax_ntk_nngp` when using DLPack to go back and forth between Jax and Torch tensors and it currently does not work.
- Currently `ntk_depth` is just hardcoded in the `jax_ntk_nngp.py` file and not actually using the argparse argument.
- Add an argparse flag to properly toggle computations of `per_class_agops` on and off for both `train_kernel.py` and `train_net.py`.
- Add `per_class_agops` computation to laplace kernel, jax_ntk_nngp kernels, and general_quadratic kernels.
- Add an argparse flag to toggle visualizing spectra of AGOPs in wandb.
