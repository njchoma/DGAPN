from rdkit.Chem.Descriptors import MolLogP

def get_logp_score(states):
    if not isinstance(states, list):
        return MolLogP(states)
    else:
        return [MolLogP(state) for state in states]
