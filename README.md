# exaLearnMol

This project aims to:

1. Generate molecules for successful interactions with covid 19 proteins, in developing molecules for a covid 19 treatment.

Final product wish list:

1. 3d representation with large drop-out

2. Retrieval system attention mechanism (done)

3. User-specified number of GNN layers to share

4. Fix multi-head (done)

5. Adversarial

## Getting Started:
1. Create conda environment
```bash
conda create -n my-mol-env python=3.8
conda activate my-mol-env
pip install wget pyyaml gym tensorboard
```
2. Install learning library
- Pytorch **1.8**.0 (https://pytorch.org/get-started/locally/)
- Pytorch Geometric **1.6**.3 (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

  \* *make sure to install the right versions for your toolkit*

3. Install chemical dependencies
```bash
conda install -c conda-forge rdkit
pip install crem
```
