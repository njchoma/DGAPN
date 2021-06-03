### AutoDock-GPU implementation in the DGAPN framework

#### Background

Our pipeline achieves high-throughput processing by taking advantage of
the performance benefits of AutoDock-GPU
(<https://arxiv.org/pdf/2007.03678.pdf>; <https://doi.org/10.26434/chemrxiv.9702389.v1>)
for structure-based molecular docking and the calculation of the binding
affinity of the putative protein-ligand complexes generated.

The recently published paper
(<https://pubs.acs.org/doi/10.1021/acs.jctc.0c01006>)
describes the latest implementation in greater detail.

Refer to the AutoDock-GPU documentation
(<https://github.com/ccsb-scripps/AutoDock-GPU/wiki>)
for more information.

**Overview of the usage of AutoDock-GPU in the DGAPN framework :**

1.  APIs / software used:
	-   AutoDockTools (<http://autodock.scripps.edu/resources/adt>)
		> To define a search box for docking on the receptor and to generate the
		> inputs for AutoGrid4.

	-   AutoGrid4
    ([[http://autodock.scripps.edu/wiki/AutoGrid)]{.ul}](http://autodock.scripps.edu/wiki/AutoGrid)
		> To pre-calculate atom-specific grid maps of energy of interaction for
		> a given protein target

	-   RDKit (<https://www.rdkit.org/>)
		> To convert, filter, and manipulate chemical data

	-   obabel
    ([[https://open-babel.readthedocs.io/en/latest/Command-line_tools/babel.html]{.ul}](https://open-babel.readthedocs.io/en/latest/Command-line_tools/babel.html))
		> To convert, filter, and manipulate chemical data

	-   AutoDock-GPU
    ([[https://github.com/ccsb-scripps/AutoDock-GPU/wiki]{.ul}](https://github.com/ccsb-scripps/AutoDock-GPU/wiki))

2.  Input: SMILES strings of all molecules generated per iteration by
    DGAPN

3.  Output: List of reward scores, which are the negative of the docking
    score returned by AutoDock-GPU (since DGAPN performs maximization)

Notes:

-   AutoDockTools and AutoGrid4 are required in the preliminary step
    that needs to be performed for each receptor in order to prepare
    inputs for targeted molecular docking, using AutoDock-GPU

-   AutoDock-GPU needs to be installed on a Linux compute server with
    GPUs

-   Executable paths must be set for the specific computing system for
    Obabel and AutoDock-GPU

**Required receptor input files:**

AutoDock-GPU requires several files specific to the receptor.

-   .map files: One for each atom type, they provide electrostatic
    potential and desolvation free energy grid maps, for the given
    target macromolecule

-   .maps.fld file: Describes the consistent set of atomic affinity grid
    maps

-   .maps.xyz file: Describes the spatial extents of the grid box

-   .pdbqt file: Contains the atomic coordinates, partial charges, and
    AutoDock atom types of the receptor

**Pre-processing structural information from the receptor:**

The receptor input files provided in these resources are for docking in
the catalytic site of NSP15. In order to create the required files for a
new receptor the following steps can be followed:

1.  Using AutoDockTools, define the search space and generate the inputs
    to run AutoGrid4

    -   Grid > Macromolecule > Open

    -   Save as pdbqt

    -   Set map types > Directly

    -   Define a list of atom types that your ligands may contain. For
        our test set, they are: A Br C Cl F HD N NA OA S SA. More
        information about this can be found at
        [[http://autodock.scripps.edu/faqs-help/faq/where-do-i-set-the-autodock-4-force-field-parameters]{.ul}](http://autodock.scripps.edu/faqs-help/faq/where-do-i-set-the-autodock-4-force-field-parameters).

    -   Grid > Grid box

    -   Use spacing = 1 Å and select the docking space

    -   File > Close saving current

    -   Grid > Output > Save GPF

	> This will generate the receptor PDBQT and .gpf files.

2.  Pre-calculate atom-specific maps of energy of interaction running
    AutoGrid4

    -   Run AutoGrid4 with: autogrid4 -p \[file\].gpf

	> This will generate the .map and .fld needed to run adt-gpu.

**Autodock-GPU Processing Workflow:**

Our implementation of AutoDock-GPU for DGAPN is summarized as follows:

1.  Using RDKit, the chemical validity of the SMILES strings of the
    generated molecules is checked. If valid, the SMILES are converted
    into PDB, using the Chem.MolToPDBBlock module. Invalid SMILES are
    given a score of 0.0

2.  PDB files are converted to PDBQT, using obabel. Both the PDB and
    PDBQT formats are temporarily saved in a /ligands/ sub-directory,
    under the work directory

3.  A list of all PDBQT ligands to be evaluated is stored in a text file
    in the /ligands/ sub-directory

4.  Molecular docking is performed using AutoDock-GPU. In this study, we
    defined 10 LGA replicas per molecule (the nrun parameter) - this is
    a default value for targeted docking

5.  Output is parsed to fetch the docking scores, the negative of the
    docking score is returned for all SMILES as reward scores. The
    reward function for DGAPN (get_dock_score) can be found at
    src/reward/adtgpu in the source code.
