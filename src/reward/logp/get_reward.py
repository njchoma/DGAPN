from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import MolLogP
from rdkit.Chem import RDConfig

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

import sascorer

def get_logp_score(states):
    if not isinstance(states, list):
        return MolLogP(states)
    else:
        return [MolLogP(state) for state in states]

def get_penalized_logp(states):
    if not isinstance(states, list):
        return penalized_logp(states)
    else:
        return [penalized_logp(state) for state in states]

# From MolDQN 
# https://github.com/caiyingchun/MolDQN/blob/cd2b738446f13cd76c6ab2d041ff03cd28e2a440/chemgraph/dqn/py/molecules.py

def get_largest_ring_size(molecule):
    """Calculates the largest ring size in the molecule.
    Refactored from
    https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
    Args:
    molecule: Chem.Mol. A molecule.
    Returns:
    Integer. The largest ring size.
    """
    cycle_list = molecule.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length

def penalized_logp(molecule):
    """Calculates the penalized logP of a molecule.
    Refactored from
    https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
    See Junction Tree Variational Autoencoder for Molecular Graph Generation
    https://arxiv.org/pdf/1802.04364.pdf
    Section 3.2
    Penalized logP is defined as:
    y(m) = logP(m) - SA(m) - cycle(m)
    y(m) is the penalized logP,
    logP(m) is the logP of a molecule,
    SA(m) is the synthetic accessibility score,
    cycle(m) is the largest ring size minus by six in the molecule.
    Args:
    molecule: Chem.Mol. A molecule.
    Returns:
    Float. The penalized logP value.
    """
    log_p = MolLogP(molecule)
    sas_score = sascorer.calculateScore(molecule)
    largest_ring_size = get_largest_ring_size(molecule)
    cycle_score = max(largest_ring_size - 6, 0)
    return log_p - sas_score - cycle_score

#print(penalized_logp(Chem.MolFromSmiles('Cc1ccccc1')))
