import os
import re
import sys
import shutil
import argparse
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

def get_dock_score(states):
    if not isinstance(states, list):
        states = [states]
    smiles = [Chem.MolToSmiles(mol) for mol in states]

    #Debugging flag
    DEBUG=False

    #Setup parameters (TODO - genearlize/simplify this)
    receptor_file="./src/reward/adtgpu/receptor/NSP15_6W01_A_3_H_receptor.pdbqt"
    smiles = str(smiles)
    run_dir="./src/reward/adtgpu/autodockgpu"
    #Executable paths
    obabel_path="/gpfs/alpine/syb105/proj-shared/Personal/manesh/BIN/openbabel/summit/build/bin/obabel"
    adt_path="/gpfs/alpine/syb105/proj-shared/Personal/gabrielgaz/Apps/summit/autoDockGPU2/bin/autodock_gpu_64wi"

    #Check that input file path exist
    if not os.path.exists(receptor_file):
        exit("Receptor file does not exist: {}".format(receptor_file))   
    #Create output dirs
    ligands_dir="/ligands"
    if not os.path.exists(run_dir): 
        os.makedirs(run_dir)
    if not os.path.exists(run_dir+ligands_dir):
        os.makedirs(run_dir+ligands_dir)

    #Parse smiles input into array
    if(DEBUG): print("Original smiles:\n{}".format(smiles))
    smiles=re.sub('\ |\'', '', smiles[1:-1]).split(",")
    if(DEBUG): print("List of smiles:\n{}".format('\n'.join(smiles)))

    #Loop over smile strings to convert to pdbqt
    ligs_list=[]
    sm_counter=1
    for smile in smiles:
        #Prepare SMILES for conversion, convert to pdb
        my_mol = Chem.MolFromSmiles(smile)
        my_mol_with_H=Chem.AddHs(my_mol)
        AllChem.EmbedMolecule(my_mol_with_H)
        AllChem.MMFFOptimizeMolecule(my_mol_with_H)
        my_embedded_mol = Chem.RemoveHs(my_mol_with_H)
        #print("Printing MolToPDBBlock:\n".format(Chem.MolToPDBBlock(my_embedded_mol))
 
        #Create temp directory needed for obabel
        tmp_file=run_dir+ligands_dir+"/ligand"+str(sm_counter)+".pdb"
        with open(tmp_file,'w') as f:
            f.write(Chem.MolToPDBBlock(my_embedded_mol))
        
        #Create name for output pdbqt file
        ligand_out=run_dir+ligands_dir+"/ligand"+str(sm_counter)+".pdbqt"

        #Convert pdb to pdbqt
        cmd=obabel_path+" --partialcharge gasteiger --addfilename -ipdb "
        cmd+=tmp_file+" -opdbqt -O "+ligand_out
        if(DEBUG): print("\nCmd to run:\n{}".format(cmd))
        if(DEBUG): subprocess.Popen(cmd,shell=True).wait()
        else: subprocess.Popen(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).wait()
        if(DEBUG): print("Done!")
    
        #Clean up and increment smile counter
        os.remove(tmp_file)
        ligand_store_file=ligand_out.split('/')[-1][:-6]
        ligs_list.append(ligand_store_file)
        sm_counter+=1
    
    #Get stub name of receptor and field file
    receptor_dir='/'.join(receptor_file.split('/')[:-1])
    receptor_stub=receptor_file.split('/')[-1][:-6] #rm .pdbqt=6
    if(DEBUG): print("\nReceptor dir:  {}".format(receptor_dir))
    if(DEBUG): print("Receptor stub: {}".format(receptor_stub))
    receptor_field=receptor_stub+".maps.fld"

    #Create run file for Autodock-gpu
    run_file=run_dir+"/ligs_list.runfile"
    run_file_lbl="ligs_list.runfile"
    with open(run_file,'w') as f:
        f.write(receptor_field+'\n')
        for lig in ligs_list:
            f.write("ligands/"+lig+".pdbqt\n")
            f.write("ligands/"+lig+'\n')

    #Copy map files to run dir
    cmd="cp "+receptor_dir+"/"+receptor_stub+"* "+run_dir
    if(DEBUG): print("\nCopy cmd to run: {}".format(cmd))
    if(DEBUG): subprocess.Popen(cmd,shell=True).wait()
    else: subprocess.Popen(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).wait()

    #Set up autodock-gpu run command
    cmd=adt_path + " -filelist "+run_file_lbl+" -nrun 10"
    if(DEBUG): print("\nAutodock cmd to run: {}".format(cmd))

    #Run autodock-gpu (in run_dir and move back)
    cur_dir=os.getcwd()
    os.chdir(run_dir)
    if(DEBUG): subprocess.Popen(cmd,shell=True).wait()
    else: subprocess.Popen(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).wait()
    os.chdir(cur_dir)

    #Read final scores into list
    pred_docking_score=[]
    for lig in ligs_list:
        #Parse for final score
        lig_path=run_dir+ligands_dir+"/"+lig+".dlg"
        if not os.path.exists(lig_path):
            print("ERROR: No such file {}".format(lig_path))
        else: 
            grep_cmd = "grep -2 \"^Rank \" "+lig_path+" | head -5 | tail -1 | cut -d \'|\' -f2 | sed \'s/ //g\'"
            grep_out=os.popen(grep_cmd).read()
            pred_docking_score.append(-float(grep_out.strip()))
            
    shutil.rmtree(run_dir)
    print("Reward Scores (-dock): {}".format(pred_docking_score))
    return (pred_docking_score)
