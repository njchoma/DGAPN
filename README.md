# Distilled Graph Attention Policy Network

This repository is the official implementation of Distilled Graph Attention Policy Network (DGAPN) in the paper [**Spatial Graph Attention and Curiosity-driven Policy for Antiviral Drug Discovery**](http://arxiv.org/abs/2106.02190).


## Installation

### 1. Set up modules
```bash
git clone https://github.com/njchoma/DGAPN
cd DGAPN
git submodule update --init --recursive
```

### 2. Create conda environment
```bash
conda create -n my-mol-env --file requirements.txt
conda activate my-mol-env
```

### 3. Install learning library
- Pytorch **1.8**.0 (https://pytorch.org/get-started/locally/)
- Pytorch Geometric **1.6**.3 (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

  \* *make sure to install the right versions for your toolkit*

### 4. Install software dependency (if docking reward is desired)

To evaluate molecular docking scores, the docking program [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU/wiki) and [Open Babel](https://open-babel.readthedocs.io/en/latest/Command-line_tools/babel.html) need to be installed. After installations, change `OBABEL_PATH` and `ADT_PATH` in [the reward function](src/reward/adtgpu/get_reward.py) to the corresponding executable paths on your system.

[The provided resources](src/reward/adtgpu/receptor) are for docking in the catalytic site of NSP15. If docking against a new protein is desired, several input receptor files need to be generated, see [the sub-directory](src/reward/adtgpu) for more details.

For optimizing logP, penalized logP, etc., this step is not necessary.


## Training

Once the conda environment and Autodock-GPU are set up, the function call to train the DGAPN is:

```bash
./main_dgapn.sh
```

A list of flags may be found in `main_dgapn.sh` and `src/main_dgapn.py` for experimentation with different network and training parameters. If you wish to produce a pre-trained graph embedding model for DGAPN training, or just want to try out supervised learning with Spatial Graph Attention Network (sGAT), check out the submodule [here](https://github.com/yulun-rayn/sGAT) (installation steps can be skipped if a DGAPN environment is already established).

## Evaluation

After training a model, use `main_eval.sh` to produce and evaluate molecules.
The flag `--policy_path` should be modified to direct to a trained DGAPN model.

```bash
./main_eval.sh
```

Molecules will be saved in the artifact directory (set via the `--artifact_path` flag in `main_eval.sh`) as a csv file, where each line contains a molecule's SMILES string and associated docking score.

## Results

Scores represent docking values as evaluated by AutoDock GPU. 

|    Model           | mean | 1st    | 2nd    | 3rd    |
| ------------------ |------|--------|--------|--------|
| Reinvent           | -5.6 | -10.22 | -9.76  | -9.50  |
| JTVAE              | -5.6 | -8.56  | -8.39  | -8.39  |  
| GCPN               | -4.8 | -16.53 | -10.72 | -10.6  |
| MolDQN             | -6.7 | -10.88 | -10.51 | -10.36 |
| DGAPN              | **-8.3** | -12.78 | -12.12 | -11.72 |
