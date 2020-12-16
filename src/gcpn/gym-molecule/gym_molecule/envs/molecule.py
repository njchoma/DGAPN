import gym
import itertools
import numpy as np
from rdkit import Chem  # TODO(Bowen): remove and just use AllChem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
# import gym_molecule
import copy
import networkx as nx
from gym_molecule.envs.sascorer import calculateScore
from gym_molecule.dataset.dataset_utils import gdb_dataset, mol_to_nx, nx_to_mol
import random
import time
import matplotlib.pyplot as plt
import csv

from crem.crem import mutate_mol, grow_mol, link_mols
from contextlib import contextmanager
import sys, os


# block std out
@contextmanager
def nostdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# TODO(Bowen): check, esp if input is not radical
def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


def load_scaffold():
    cwd = os.path.dirname(__file__)
    path = os.path.join(os.path.dirname(cwd), 'dataset',
                        'vocab.txt')  # gdb 13
    with open(path, 'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data = [Chem.MolFromSmiles(row[0]) for row in reader]
        data = [mol for mol in data if mol.GetRingInfo().NumRings() == 1 and (
                    mol.GetRingInfo().IsAtomInRingOfSize(0, 5) or mol.GetRingInfo().IsAtomInRingOfSize(0, 6))]
        for mol in data:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        print('num of scaffolds:', len(data))
        return data


def load_conditional(filepath):
    import csv
    with open(filepath, 'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        # List of lists, [[Smile, Score, row_number],...,]
        data = [[row[1], row[0], id] for id, row in enumerate(reader)]
        data = data[0:800]
    return data


class MoleculeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def init(self, data_type='zinc', logp_ratio=1, qed_ratio=1, sa_ratio=1, reward_step_total=1, is_normalize=0,
             reward_type='gan', reward_target=0.5, has_scaffold=False, has_feature=False, is_conditional=False,
             conditional='', max_action=128, min_action=20, force_final=False):
        '''
        own init function, since gym does not support passing argument
        '''
        self.is_normalize = bool(is_normalize)
        self.is_conditional = is_conditional
        self.has_feature = has_feature
        self.reward_type = reward_type
        self.reward_target = reward_target
        self.force_final = force_final

        if self.is_conditional:
            self.conditional_list = load_conditional(conditional)
            self.conditional = random.sample(self.conditional_list, 1)[0]
            self.mol = Chem.RWMol(Chem.MolFromSmiles(self.conditional[0]))
            #Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        else:
            self.mol = Chem.RWMol()
        self.smile_list = []
        if data_type == 'gdb':
            possible_atoms = ['C', 'N', 'O', 'S', 'Cl']  # gdb 13
        elif data_type == 'zinc':
            possible_atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl',
                              'Br']  # ZINC
        if self.has_feature:
            self.possible_formal_charge = np.array([-1, 0, 1])
            self.possible_implicit_valence = np.array([-1, 0, 1, 2, 3, 4])
            self.possible_ring_atom = np.array([True, False])
            self.possible_degree = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            self.possible_hybridization = np.array([
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2],
                dtype=object)
        possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE]  # , Chem.rdchem.BondType.AROMATIC
        self.atom_type_num = len(possible_atoms)
        self.possible_atom_types = np.array(possible_atoms)
        self.possible_bond_types = np.array(possible_bonds, dtype=object)

        if self.has_feature:
            # self.d_n = len(self.possible_atom_types) + len(
            #     self.possible_formal_charge) + len(
            #     self.possible_implicit_valence) + len(self.possible_ring_atom) + \
            #       len(self.possible_degree) + len(self.possible_hybridization)
            self.d_n = len(self.possible_atom_types) + 6  # 6 is the ring feature
        else:
            self.d_n = len(self.possible_atom_types)

        self.max_action = max_action
        self.min_action = min_action
        if data_type == 'gdb':
            self.max_atom = 13 + len(possible_atoms)  # gdb 13
        elif data_type == 'zinc':
            if self.is_conditional:
                self.max_atom = 38 + len(possible_atoms) + self.min_action  # ZINC
            else:
                self.max_atom = 38 + len(possible_atoms)  # ZINC  + self.min_action

        self.logp_ratio = logp_ratio
        self.qed_ratio = qed_ratio
        self.sa_ratio = sa_ratio
        self.reward_step_total = reward_step_total
        self.action_space = gym.spaces.MultiDiscrete([self.max_atom, self.max_atom, 3, 2])
        self.observation_space = {}
        self.observation_space['adj'] = gym.Space(shape=[len(possible_bonds), self.max_atom, self.max_atom])
        self.observation_space['node'] = gym.Space(shape=[1, self.max_atom, self.d_n])

        self.counter = 0

        ## load expert data
        cwd = os.path.dirname(__file__)
        if data_type == 'gdb':
            path = os.path.join(os.path.dirname(cwd), 'dataset',
                                'gdb13.rand1M.smi.gz')  # gdb 13
        elif data_type == 'zinc':
            path = os.path.join(os.path.dirname(cwd), 'dataset',
                                '250k_rndm_zinc_drugs_clean_sorted.smi')  # ZINC
        self.dataset = gdb_dataset(path)

        ## load scaffold data if necessary
        self.has_scaffold = has_scaffold
        if has_scaffold:
            self.scaffold = load_scaffold()
            self.max_scaffold = 6

        self.level = 0  # for curriculum learning, level starts with 0, and increase afterwards

    def level_up(self):
        self.level += 1

    def seed(self, seed):
        np.random.seed(seed=seed)
        random.seed(seed)

    def normalize_adj(self, adj):
        degrees = np.sum(adj, axis=2)
        # print('degrees',degrees)
        D = np.zeros((adj.shape[0], adj.shape[1], adj.shape[2]))
        for i in range(D.shape[0]):
            D[i, :, :] = np.diag(np.power(degrees[i, :], -0.5))
        adj_normal = D @ adj @ D
        adj_normal[np.isnan(adj_normal)] = 0
        return adj_normal

    # TODO(Bowen): The top try, except clause allows error messages from step
    # to be printed when running run_molecules.py. For debugging only
    def step(self, action, memory, crem=False):
        """
        Perform a given action
        :param action:
        :param memory:
        :param crem:
        :return: reward of 1 if resulting molecule graph does not exceed valency,
        -1 if otherwise
        """
        ### init
        info = {}  # info we care about
        self.mol_old = copy.deepcopy(self.mol)  # keep old mol
        total_atoms = self.mol.GetNumAtoms()

        if crem:
            if action == -1:
                stop = True
            else:
                self.mol = memory.states[-1][action]
                #Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                stop = False

        ### take non crem action
        elif action[0, 3] == 0 or self.counter < self.min_action:  # not stop
            stop = False
            if action[0, 1] >= total_atoms:
                self._add_atom(action[0, 1] - total_atoms)  # add new node
                action[0, 1] = total_atoms  # new node id
                self._add_bond(action)  # add new edge
            else:
                self._add_bond(action)  # add new edge
        else:  # stop
            stop = True

        ### calculate intermediate rewards
        reward_step = 0
        # if self.check_valency():
        #     if self.mol.GetNumAtoms() + self.mol.GetNumBonds() - self.mol_old.GetNumAtoms() - self.mol_old.GetNumBonds() > 0:
        #         reward_step = self.reward_step_total / self.max_atom  # successfully add node/edge
        #         self.smile_list.append(self.get_final_smiles())
        #     else:
        #         reward_step = -self.reward_step_total / self.max_atom  # edge exist
        # else:
        #     reward_step = -self.reward_step_total / self.max_atom  # invalid action
        #     self.mol = self.mol_old

        ### calculate terminal rewards
        # todo: add terminal action

        if self.is_conditional:
            terminate_condition = (self.mol.GetNumAtoms() >= self.max_atom - self.possible_atom_types.shape[
                0] - self.min_action or self.counter >= self.max_action or stop) and self.counter >= self.min_action
        else:
            terminate_condition = (self.mol.GetNumAtoms() >= self.max_atom - self.possible_atom_types.shape[
                0] or self.counter >= self.max_action or stop) and self.counter >= self.min_action
        if terminate_condition or self.force_final:
            # default reward
            reward_valid = 2
            reward_qed = 0
            reward_sa = 0
            reward_logp = 0
            reward_final = 0
            flag_steric_strain_filter = True
            flag_zinc_molecule_filter = True

            if not self.check_chemical_validity():
                reward_valid -= 5
            else:
                # final mol object where any radical electrons are changed to bonds to hydrogen
                final_mol = self.get_final_mol()
                #s = Chem.MolToSmiles(final_mol, isomericSmiles=True)
                #final_mol = Chem.MolFromSmiles(s)

                # mol filters with negative rewards
                if not steric_strain_filter(final_mol):  # passes 3D conversion, no excessive strain
                    reward_valid -= 1
                    flag_steric_strain_filter = False
                if not zinc_molecule_filter(final_mol):  # does not contain any problematic functional groups
                    reward_valid -= 1
                    flag_zinc_molecule_filter = False

                # property rewards
                try:
                    # 1. QED reward. Can have values [0, 1]. Higher the better
                    reward_qed += qed(final_mol) * self.qed_ratio
                    # 2. Synthetic accessibility reward. Values naively normalized to [0, 1]. Higher the better
                    sa = -1 * calculateScore(final_mol)
                    reward_sa += (sa + 10) / (10 - 1) * self.sa_ratio
                    # 3. Logp reward. Higher the better
                    # reward_logp += MolLogP(self.mol)/10 * self.logp_ratio
                    reward_logp += reward_penalized_log_p(final_mol) * self.logp_ratio
                    if self.reward_type == 'logppen':
                        reward_final += reward_penalized_log_p(final_mol) / 3
                    elif self.reward_type == 'logp_target':
                        # reward_final += reward_target(final_mol,target=self.reward_target,ratio=0.5,val_max=2,val_min=-2,func=MolLogP)
                        # reward_final += reward_target_logp(final_mol,target=self.reward_target)
                        reward_final += reward_target_new(final_mol, MolLogP, x_start=self.reward_target,
                                                          x_mid=self.reward_target + 0.25)
                    elif self.reward_type == 'qed':
                        reward_final += reward_qed * 2
                    elif self.reward_type == 'qedsa':
                        reward_final += (reward_qed * 1.5 + reward_sa * 0.5)
                    elif self.reward_type == 'qed_target':
                        # reward_final += reward_target(final_mol,target=self.reward_target,ratio=0.1,val_max=2,val_min=-2,func=qed)
                        reward_final += reward_target_qed(final_mol, target=self.reward_target)
                    elif self.reward_type == 'mw_target':
                        # reward_final += reward_target(final_mol,target=self.reward_target,ratio=40,val_max=2,val_min=-2,func=rdMolDescriptors.CalcExactMolWt)
                        # reward_final += reward_target_mw(final_mol,target=self.reward_target)
                        reward_final += reward_target_new(final_mol, rdMolDescriptors.CalcExactMolWt,
                                                          x_start=self.reward_target, x_mid=self.reward_target + 25)


                    elif self.reward_type == 'gan':
                        reward_final = 0
                    else:
                        print('reward error!')
                        reward_final = 0



                except:  # if any property reward error, reset all
                    print('reward error')

            new = True  # end of episode
            if self.force_final:
                reward = reward_final
            else:
                reward = reward_step + reward_valid + reward_final
            info['smile'] = self.get_final_smiles()
            info['reward_valid'] = reward_valid
            info['reward_qed'] = reward_qed
            info['reward_sa'] = reward_sa
            info['final_stat'] = reward_final
            info['reward'] = reward
            info['flag_steric_strain_filter'] = flag_steric_strain_filter
            info['flag_zinc_molecule_filter'] = flag_zinc_molecule_filter
            info['stop'] = stop

            if self.is_conditional:
                info['start_smile'] = Chem.MolToSmiles(
                    convert_radical_electrons_to_hydrogens(Chem.MolFromSmiles(self.conditional[0])), \
                    isomericSmiles=True)
        ### use stepwise reward
        else:
            new = False
            # print('counter', self.counter, 'new', new, 'reward_step', reward_step)
            reward = reward_step

        # get observation
        ob = self.get_observation()

        self.counter += 1
        if new:
            self.counter = 0

        return ob, reward, new, info

    def reset(self, smile=None):
        '''
        to avoid error, assume an atom already exists
        :return: ob
        '''
        if self.is_conditional:
            self.conditional = random.sample(self.conditional_list, 1)[0]
            self.mol = Chem.RWMol(Chem.MolFromSmiles(self.conditional[0]))
            #Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        elif smile is not None:
            self.mol = Chem.RWMol(Chem.MolFromSmiles(smile))
            #Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        else:
            self.mol = Chem.RWMol()
            # self._add_atom(np.random.randint(len(self.possible_atom_types)))  # random add one atom
            self._add_atom(0)  # always add carbon first
        self.smile_list = [self.get_final_smiles()]
        self.counter = 0
        ob = self.get_observation()
        return ob

    def render(self, mode='human', close=False):
        return

    def _add_atom(self, atom_type_id):
        """
        Adds an atom
        :param atom_type_id: atom_type id
        :return:
        """
        # assert action.shape == (len(self.possible_atom_types),)
        # atom_type_idx = np.argmax(action)
        atom_symbol = self.possible_atom_types[atom_type_id]
        self.mol.AddAtom(Chem.Atom(atom_symbol))

    def _add_bond(self, action):
        '''

        :param action: [first_node, second_node, bong_type_id]
        :return:
        '''
        # GetBondBetweenAtoms fails for np.int64
        bond_type = self.possible_bond_types[action[0, 2]]

        # if bond exists between current atom and other atom, modify the bond
        # type to new bond type. Otherwise create bond between current atom and
        # other atom with the new bond type
        bond = self.mol.GetBondBetweenAtoms(int(action[0, 0]), int(action[0, 1]))
        if bond:
            # print('bond exist!')
            return False
        else:
            self.mol.AddBond(int(action[0, 0]), int(action[0, 1]), order=bond_type)
            # bond = self.mol.GetBondBetweenAtoms(int(action[0, 0]), int(action[0, 1]))
            # bond.SetIntProp('ordering',self.mol.GetNumBonds())
            return True

    def _modify_bond(self, action):
        """
        Adds or modifies a bond (currently no deletion is allowed)
        :param action: np array of dim N-1 x d_e, where N is the current total
        number of atoms, d_e is the number of bond types
        :return:
        """
        assert action.shape == (self.current_atom_idx, len(self.possible_bond_types))
        other_atom_idx = int(np.argmax(action.sum(axis=1)))  # b/c
        # GetBondBetweenAtoms fails for np.int64
        bond_type_idx = np.argmax(action.sum(axis=0))
        bond_type = self.possible_bond_types[bond_type_idx]

        # if bond exists between current atom and other atom, modify the bond
        # type to new bond type. Otherwise create bond between current atom and
        # other atom with the new bond type
        bond = self.mol.GetBondBetweenAtoms(self.current_atom_idx, other_atom_idx)
        if bond:
            bond.SetBondType(bond_type)
        else:
            self.mol.AddBond(self.current_atom_idx, other_atom_idx, order=bond_type)
            self.total_bonds += 1

    def get_num_atoms(self):
        return self.total_atoms

    def get_num_bonds(self):
        return self.total_bonds

    def check_chemical_validity(self):
        """
        Checks the chemical validity of the mol object. Existing mol object is
        not modified. Radicals pass this test.
        :return: True if chemically valid, False otherwise
        """
        s = Chem.MolToSmiles(self.mol, isomericSmiles=True)
        m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
        if m:
            return True
        else:
            return False

    def check_valency(self):
        """
        Checks that no atoms in the mol have exceeded their possible
        valency
        :return: True if no valency issues, False otherwise
        """
        try:
            Chem.SanitizeMol(self.mol,
                             sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return True
        except ValueError:
            return False

    # TODO(Bowen): check if need to sanitize again
    def get_final_smiles(self):
        """
        Returns a SMILES of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = convert_radical_electrons_to_hydrogens(self.mol)
        return Chem.MolToSmiles(m, isomericSmiles=True)

    # TODO(Bowen): check if need to sanitize again
    def get_final_mol(self):
        """
        Returns a rdkit mol object of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = convert_radical_electrons_to_hydrogens(self.mol)
        return m

    def get_observation(self):
        """
        ob['adj']:d_e*n*n --- 'E'
        ob['node']:1*n*d_n --- 'F'
        n = atom_num + atom_type_num
        """
        mol = copy.deepcopy(self.mol)
        return self.mol


### YES/NO filters ###
def zinc_molecule_filter(mol):
    """
    Flags molecules based on problematic functional groups as
    provided set of ZINC rules from
    http://blaster.docking.org/filtering/rules_default.txt.
    :param mol: rdkit mol object
    :return: Returns True if molecule is okay (ie does not match any of
    therules), False if otherwise
    """
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
    catalog = FilterCatalog(params)
    return not catalog.HasMatch(mol)


# TODO(Bowen): check
def steric_strain_filter(mol, cutoff=0.82,
                         max_attempts_embed=20,
                         max_num_iters=200):
    """
    Flags molecules based on a steric energy cutoff after max_num_iters
    iterations of MMFF94 forcefield minimization. Cutoff is based on average
    angle bend strain energy of molecule
    :param mol: rdkit mol object
    :param cutoff: kcal/mol per angle . If minimized energy is above this
    threshold, then molecule fails the steric strain filter
    :param max_attempts_embed: number of attempts to generate initial 3d
    coordinates
    :param max_num_iters: number of iterations of forcefield minimization
    :return: True if molecule could be successfully minimized, and resulting
    energy is below cutoff, otherwise False
    """
    # check for the trivial cases of a single atom or only 2 atoms, in which
    # case there is no angle bend strain energy (as there are no angles!)
    if mol.GetNumAtoms() <= 2:
        return True

    # make copy of input mol and add hydrogens
    m = copy.deepcopy(mol)
    m_h = Chem.AddHs(m)

    # generate an initial 3d conformer
    try:
        flag = AllChem.EmbedMolecule(m_h, maxAttempts=max_attempts_embed)
        if flag == -1:
            # print("Unable to generate 3d conformer")
            return False
    except:  # to catch error caused by molecules such as C=[SH]1=C2OC21ON(N)OC(=O)NO
        # print("Unable to generate 3d conformer")
        return False

    # set up the forcefield
    AllChem.MMFFSanitizeMolecule(m_h)
    if AllChem.MMFFHasAllMoleculeParams(m_h):
        mmff_props = AllChem.MMFFGetMoleculeProperties(m_h)
        try:  # to deal with molecules such as CNN1NS23(=C4C5=C2C(=C53)N4Cl)S1
            ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
        except:
            # print("Unable to get forcefield or sanitization error")
            return False
    else:
        # print("Unrecognized atom type")
        return False

    # minimize steric energy
    try:
        ff.Minimize(maxIts=max_num_iters)
    except:
        # print("Minimization error")
        return False

    # ### debug ###
    # min_e = ff.CalcEnergy()
    # print("Minimized energy: {}".format(min_e))
    # ### debug ###

    # get the angle bend term contribution to the total molecule strain energy
    mmff_props.SetMMFFBondTerm(False)
    mmff_props.SetMMFFAngleTerm(True)
    mmff_props.SetMMFFStretchBendTerm(False)
    mmff_props.SetMMFFOopTerm(False)
    mmff_props.SetMMFFTorsionTerm(False)
    mmff_props.SetMMFFVdWTerm(False)
    mmff_props.SetMMFFEleTerm(False)

    ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)

    min_angle_e = ff.CalcEnergy()
    # print("Minimized angle bend energy: {}".format(min_angle_e))

    # find number of angles in molecule
    # TODO(Bowen): there must be a better way to get a list of all angles
    # from molecule... This is too hacky
    num_atoms = m_h.GetNumAtoms()
    atom_indices = range(num_atoms)
    angle_atom_triplets = itertools.permutations(atom_indices, 3)  # get all
    # possible 3 atom indices groups. Currently, each angle is represented by
    #  2 duplicate groups. Should remove duplicates here to be more efficient
    double_num_angles = 0
    for triplet in list(angle_atom_triplets):
        if mmff_props.GetMMFFAngleBendParams(m_h, *triplet):
            double_num_angles += 1
    num_angles = double_num_angles / 2  # account for duplicate angles

    # print("Number of angles: {}".format(num_angles))

    avr_angle_e = min_angle_e / num_angles

    # print("Average minimized angle bend energy: {}".format(avr_angle_e))

    # ### debug ###
    # for i in range(7):
    #     termList = [['BondStretch', False], ['AngleBend', False],
    #                 ['StretchBend', False], ['OopBend', False],
    #                 ['Torsion', False],
    #                 ['VdW', False], ['Electrostatic', False]]
    #     termList[i][1] = True
    #     mmff_props.SetMMFFBondTerm(termList[0][1])
    #     mmff_props.SetMMFFAngleTerm(termList[1][1])
    #     mmff_props.SetMMFFStretchBendTerm(termList[2][1])
    #     mmff_props.SetMMFFOopTerm(termList[3][1])
    #     mmff_props.SetMMFFTorsionTerm(termList[4][1])
    #     mmff_props.SetMMFFVdWTerm(termList[5][1])
    #     mmff_props.SetMMFFEleTerm(termList[6][1])
    #     ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
    #     print('{0:>16s} energy: {1:12.4f} kcal/mol'.format(termList[i][0],
    #                                                  ff.CalcEnergy()))
    # ## end debug ###

    if avr_angle_e < cutoff:
        return True
    else:
        return False


### TARGET VALUE REWARDS ###

def reward_target(mol, target, ratio, val_max, val_min, func):
    x = func(mol)
    reward = max(-1 * np.abs((x - target) / ratio) + val_max, val_min)
    return reward


def reward_target_new(mol, func, r_max1=4, r_max2=2.25, r_mid=2, r_min=-2, x_start=500, x_mid=525):
    x = func(mol)
    return max((r_max1 - r_mid) / (x_start - x_mid) * np.abs(x - x_mid) + r_max1,
               (r_max2 - r_mid) / (x_start - x_mid) * np.abs(x - x_mid) + r_max2, r_min)


def reward_target_logp(mol, target, ratio=0.5, max=4):
    """
    Reward for a target log p
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = MolLogP(mol)
    reward = -1 * np.abs((x - target) / ratio) + max
    return reward


def reward_target_penalizelogp(mol, target, ratio=3, max=4):
    """
    Reward for a target log p
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = reward_penalized_log_p(mol)
    reward = -1 * np.abs((x - target) / ratio) + max
    return reward


def reward_target_qed(mol, target, ratio=0.1, max=4):
    """
    Reward for a target log p
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = qed(mol)
    reward = -1 * np.abs((x - target) / ratio) + max
    return reward


def reward_target_mw(mol, target, ratio=40, max=4):
    """
    Reward for a target molecular weight
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = rdMolDescriptors.CalcExactMolWt(mol)
    reward = -1 * np.abs((x - target) / ratio) + max
    return reward


# TODO(Bowen): num rings is a discrete variable, so what is the best way to
# calculate the reward?
def reward_target_num_rings(mol, target):
    """
    Reward for a target number of rings
    :param mol: rdkit mol object
    :param target: int
    :return: float (-inf, 1]
    """
    x = rdMolDescriptors.CalcNumRings(mol)
    reward = -1 * (x - target) ** 2 + 1
    return reward


# TODO(Bowen): more efficient if we precalculate the target fingerprint
from rdkit import DataStructs


def reward_target_molecule_similarity(mol, target, radius=2, nBits=2048,
                                      useChirality=True):
    """
    Reward for a target molecule similarity, based on tanimoto similarity
    between the ECFP fingerprints of the x molecule and target molecule
    :param mol: rdkit mol object
    :param target: rdkit mol object
    :return: float, [0.0, 1.0]
    """
    x = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius,
                                                       nBits=nBits,
                                                       useChirality=useChirality)
    target = rdMolDescriptors.GetMorganFingerprintAsBitVect(target,
                                                            radius=radius,
                                                            nBits=nBits,
                                                            useChirality=useChirality)
    return DataStructs.TanimotoSimilarity(x, target)


### TERMINAL VALUE REWARDS ###

def reward_penalized_log_p(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


# # TEST compare with junction tree paper examples from Figure 7
# assert round(reward_penalized_log_p(Chem.MolFromSmiles('ClC1=CC=C2C(C=C(C('
#                                                        'C)=O)C(C(NC3=CC(NC('
#                                                        'NC4=CC(C5=C('
#                                                        'C)C=CC=C5)=CC=C4)=O)=CC=C3)=O)=C2)=C1')), 2) == 5.30
# assert round(reward_penalized_log_p(Chem.MolFromSmiles('CC(NC1=CC(C2=CC=CC('
#                                                        'NC(NC3=CC=CC(C4=CC('
#                                                        'F)=CC=C4)=C3)=O)=C2)=CC=C1)=O')), 2) == 4.49
# assert round(reward_penalized_log_p(Chem.MolFromSmiles('ClC(C('
#                                                        'Cl)=C1)=CC=C1NC2=CC=CC=C2C(NC(NC3=C(C(NC4=C(Cl)C=CC=C4)=S)C=CC=C3)=O)=O')), 2) == 4.93


# smile = 'C'*38
# smile = 'CCCCCCCCCC(CCC)(CCCCCCC)CCCCCCCCC(CCCCC)CC(C)C'
# print(smile, reward_penalized_log_p(Chem.MolFromSmiles(smile)))

# if __name__ == '__main__':
#     env = gym.make('molecule-v0') # in gym format
#     # env = GraphEnv()
#     # env.init(has_scaffold=True)
#
#     ## debug
#     m_env = MoleculeEnv()
#     m_env.init(data_type='zinc',has_feature=True,is_conditional=True)
#
#
