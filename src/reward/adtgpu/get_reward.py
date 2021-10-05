import os
import re
import sys
import shutil
import argparse
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

########## Executable paths ##########

# For exaLearn systems
OBABEL_PATH = "/usr/bin/obabel"
ADT_PATH = "/clusterfs/csdata/pkg/autodock-gpu/AutoDock-GPU/bin/autodock_gpu_64wi"
# For Summit systems
#OBABEL_PATH = "/gpfs/alpine/syb105/proj-shared/Personal/manesh/BIN/openbabel/summit/build/bin/obabel"
#ADT_PATH = "/gpfs/alpine/syb105/proj-shared/Personal/gabrielgaz/Apps/summit/autoDockGPU2/bin/autodock_gpu_64wi"

########### receptor files ###########

# NSP15, site A3H
RECEPTOR_FILE = "NSP15_6W01_A_3_H_receptor.pdbqt"

######################################

def get_dock_score(states, args=None):

    #Debugging flag
    DEBUG=False

    if not isinstance(states, list):
        states = [states]
    smiles = [Chem.MolToSmiles(mol) for mol in states]
    smile_count=len(smiles)
    if(DEBUG): print("Number of smiles to score: {}".format(smile_count))

    #Setup parameters
    if(args and args.obabel_path!=''): obabel_path=args.obabel_path
    else: obabel_path=OBABEL_PATH
    if(args and args.adt_path!=''): adt_path=args.adt_path
    else: adt_path=ADT_PATH
    if(args and args.receptor_file!=''): receptor_file="./src/reward/adtgpu/receptor/"+args.receptor_file
    else: receptor_file="./src/reward/adtgpu/receptor/"+RECEPTOR_FILE
    if(args and args.run_id!=''): run_dir="./src/reward/adtgpu/autodockgpu"+str(args.run_id)
    else: run_dir="./src/reward/adtgpu/autodockgpu"
    if(DEBUG): print("adttmp: {}".format(run_dir))

    smiles = str(smiles)

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
        if(DEBUG): print("Processing: {}".format(smile))
        VALID=True
        #Verify SMILES is valid
        my_mol = Chem.MolFromSmiles(smile,sanitize=False)
        if my_mol is None:
            print("invalid SMILES: {}".format(smiles))
            VALID=False
        else:
            try:
                Chem.SanitizeMol(my_mol)
            except:
                print("invalid chemistry: {}".format(smile))
                VALID=False

        if(VALID):
            try: 
                #Prepare SMILES for conversion, convert to pdb
                #my_mol = Chem.MolFromSmiles(smile)
                my_mol_with_H=Chem.AddHs(my_mol)
                AllChem.EmbedMolecule(my_mol_with_H)
                AllChem.MMFFOptimizeMolecule(my_mol_with_H)
                my_embedded_mol = Chem.RemoveHs(my_mol_with_H)
                #print("Printing MolToPDBBlock:\n".format(Chem.MolToPDBBlock(my_embedded_mol))
            except:
                print("other SMILES error: {}".format(smile))
                VALID=False   
 
        if(VALID): 
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
        else: #invalid SMILES
            ligs_list.append(None)
        sm_counter+=1
    if(DEBUG): print("ligs_list:\n{}".format(ligs_list))

    pred_docking_score=[]
    if(len(ligs_list)>0 and not all(x==None for x in ligs_list)):#TODO refactor this
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
                if lig != None:
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
        for lig in ligs_list:
            if lig != None:
                #Parse for final score
                lig_path=run_dir+ligands_dir+"/"+lig+".dlg"
                if not os.path.exists(lig_path):
                    print("ERROR: No such file {}".format(lig_path))
                    pred_docking_score.append(0.0)
                else: 
                    grep_cmd = "grep -2 \"^Rank \" "+lig_path+" | head -5 | tail -1 | cut -d \'|\' -f2 | sed \'s/ //g\'"
                    grep_out=os.popen(grep_cmd).read()
                    pred_docking_score.append(-float(grep_out.strip()))
            else:#invalid SMILES
                pred_docking_score.append(0.00)
    else:#ligs list is empty
        print("Warning: ligs_list is empty or all None, zeroing all scores...")
        for s in range(0,smile_count):
            pred_docking_score.append(0.00)

    shutil.rmtree(run_dir, ignore_errors=True)
    if(DEBUG): print("Reward Scores (-dock): {}".format(pred_docking_score))
    return (pred_docking_score)
