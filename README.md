# Spatial Graph Attention and Curiosity-driven Policy for Antiviral Drug Discovery

This repository is the official implementation of Spatial Graph Attention Network (sGAT) and Distilled Graph Attention Policy Network (DGAPN) in the paper "Spatial Graph Attention and Curiosity-driven Policy for Antiviral Drug Discovery".


## Installation

### Conda Environment

1. Create conda environment
```bash
conda create -n my-mol-env --file requirements.txt
conda activate my-mol-env
```

2. Install learning library
- Pytorch **1.8**.0 (https://pytorch.org/get-started/locally/)
- Pytorch Geometric **1.6**.3 (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

  \* *make sure to install the right versions for your toolkit*

3. Install software dependency (if docking reward is desired)

To evaluate molecular docking scores, the docking program [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU/wiki) and [Open Babel](https://open-babel.readthedocs.io/en/latest/Command-line_tools/babel.html) need to be installed. After installations, change `OBABEL_PATH` and `ADT_PATH` in [the reward function](src/reward/adtgpu/get_reward.py) to the corresponding executable paths on your system.

[The provided resources](src/reward/adtgpu/receptor) are for docking in the catalytic site of NSP15. If docking against a new protein is desired, several input receptor files need to be generated, see [the sub-directory](src/reward/adtgpu) for more details.

For optimizing logP, penalized logP, etc., this step is not necessary.


## Run

#### Train

Once the conda environment and Autodock-GPU are set up, the function call to train DGAPN is:

```bash
./main_dgapn.sh
```

A list of flags may be found in `main_dgapn.sh` and `src/main_dgapn.py` for experimentation with different network and training parameters. If you prefer to train DGAPN with pre-trained graph embeddings or just want to try out supervised learning with sGAT, run the following to produce a predictive model:

```bash
./main_embed.sh
```

#### Evaluate

After training a model, use `main_eval.sh` to produce and evaluate molecules.
The flag `--policy_path` should be modified to direct to a trained DGAPN model.

```bash
./main_eval.sh
```

Molecules will be saved in the artifact directory (set via the `--artifact_path` flag in `main_eval.sh`) as a csv file, where each line contains a molecule's SMILES string and associated docking score.
