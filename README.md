# exaLearnMol

This project aims to:

1. Replicate [Graph Convolutional Policy Networks](https://arxiv.org/abs/1806.02473) (github [here](https://github.com/bowenliu16/rl_graph_generation)) on the zinc250k dataset.
2. Generate molecules for successful interactions with covid 19 proteins, in developing molecules for a covid 19 treatment.

## Getting Started:
1. Install rdkit.
```bash
conda create -c rdkit -n my-rdkit-env rdkit python=3.6
```
2. Install customized molecule gym environment (from https://github.com/bowenliu16/rl_graph_generation):
```bash
cd src/gcpn/gym-molecule
pip install -e .
```
3. Install other dependencies
```bash
conda install yaml mpi4py
pip install wget networkx
```
For `Pytorch` and `Pytorch Geometric`, make sure to install the right versions for your toolkit:
- Pytorch 1.7.0 (https://pytorch.org/get-started/locally/)
- Pytorch Geometric 1.6.1 (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
