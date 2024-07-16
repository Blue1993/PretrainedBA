import os
import json
import pickle
import msgpack
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

def main():
    
    #################
    # 1. Zinc dataset
    #################
    data_smiles_list, data_list = list(), list()
    atom_slices, edge_slices = [0], [0]
    total_eigvecs, total_eigvals = list(), list()
    all_atom_features, all_edge_features = list(), list()
    edge_indices = list()
    total_n_atoms = list()
    
    total_atoms, total_edges = 0, 0
    avg_degree = 0
    
    zinc_input_df = pd.read_csv(f"../data/pre-training/compound/zinc_combined_apr_8_2019.csv.gz", 
                    sep=',', compression='gzip',dtype='str')
    
    zinc_smiles_list = list(zinc_input_df["smiles"])
    zinc_id_list = list(zinc_input_df["zinc_id"])
    
    count = 0
    for i in range(len(zinc_smiles_list)):
        if i % 500 == 0:
            print(i, len(data_smiles_list), len(zinc_smiles_list))

        s = zinc_smiles_list[i]

        try:
            mol = AllChem.MolFromSmiles(s)
            if mol != None: # Ignore invalid mol objects
                n_atoms = len(mol.GetAtoms())
                
                atom_features_list, edge_index, edge_features, n_edges = get_mol_features(mol)

                all_atom_features.append(torch.tensor(atom_features_list, dtype=torch.long))
                
                avg_degree += (n_edges / 2) / n_atoms
                edge_indices.append(edge_index)
                all_edge_features.append(edge_features)
                
                total_edges += n_edges
                total_atoms += n_atoms
                total_n_atoms.append(n_atoms)
                
                edge_slices.append(total_edges)
                atom_slices.append(total_atoms)
                
                id = int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
                data_list.append(id)
                data_smiles_list.append(s)
    
        except:
            pass

    data_smiles_series = pd.Series(data_smiles_list)
    data_smiles_series.to_csv(f"../data/pre-training/compound/smilesnonH.csv", 
                index = False, header=False) 
                
    data_dict = {'mol_ids':data_list,
                 'n_atoms':torch.tensor(total_n_atoms, dtype=torch.long),
                 'atom_slices':torch.tensor(atom_slices, dtype=torch.long),
                 'edge_slices':torch.tensor(edge_slices, dtype=torch.long),
                 'edge_indices':torch.cat(edge_indices, dim=1),
                 'atom_features':torch.cat(all_atom_features, dim=0),
                 'edge_features':torch.cat(all_edge_features, dim=0),
                 'avg_degree':avg_degree / len(data_list)}
    
    torch.save(data_dict, f"../data/pre-training/compound/MgraphDatanonH.pt")

    
    #########################
    # 2. Pre-training dataset
    #########################
    Pretraining_df = pd.read_csv("../data/pre-training/interaction/interaction_sites_predictor_training_data.tsv", sep = "\t")
    Pretraining_ligand_codes = Pretraining_df.loc[:, "LigandCodes"]
    Pretraining_ligand_codes = list(set(Pretraining_ligand_codes))
    print(f"[Pretraining dataset] total unique compounds: {len(Pretraining_ligand_codes)}")
    print()
    
    # Get compound features
    atom_slices, edge_slices = [0], [0]
    all_atom_features, all_edge_features = list(), list()
    edge_indices, total_n_atoms, id_list = list(), list(), list()

    total_atoms, total_edges = 0, 0
    avg_degree = 0
    
    path = "../data/compound_raw_graph/"
    for mol_idx, lig_code in tqdm(enumerate(Pretraining_ligand_codes)):
        if mol_idx % 500 == 0:
            print(mol_idx, len(Pretraining_ligand_codes))
        sdf_path = os.path.join(path, f"{lig_code}_ideal.sdf")
        suppl = Chem.SDMolSupplier(sdf_path, removeHs = True)

        mol = next(iter(suppl))

        n_atoms = len(mol.GetAtoms())
        
        atom_features_list, edge_index, edge_features, n_edges = get_mol_features(mol)

        if lig_code in ["313", "M2T"]:
            n_atoms, atom_features_list, edge_features, edge_index, n_edges = remove_hydrogen(atom_features_list, edge_index, edge_features)

        all_atom_features.append(torch.tensor(atom_features_list, dtype=torch.long))

        if atom_features_list == 0:
            print(sdf_path)
            return 

        avg_degree += (n_edges / 2) / n_atoms 
        edge_indices.append(edge_index)
        all_edge_features.append(edge_features)
        
        total_edges += n_edges
        total_atoms += n_atoms
        total_n_atoms.append(n_atoms)
        
        edge_slices.append(total_edges)
        atom_slices.append(total_atoms)
        
        id_list.append(lig_code)

    data_dict = {'mol_ids':id_list,
                 'n_atoms':torch.tensor(total_n_atoms, dtype=torch.long),
                 'atom_slices':torch.tensor(atom_slices, dtype=torch.long),
                 'edge_slices':torch.tensor(edge_slices, dtype=torch.long),
                 'edge_indices':torch.cat(edge_indices, dim=1),
                 'atom_features':torch.cat(all_atom_features, dim=0),
                 'edge_features':torch.cat(all_edge_features, dim=0),
                 'avg_degree':avg_degree / len(id_list)}

    torch.save(data_dict, f"../data/pre-training/interaction/graph_data.pt")
    
    ##############################
    # 3. Binding affinity datasets
    ##############################
    PDBbind_df = pd.read_csv("../data/affinity/training_data.tsv", sep = "\t")
    PDBbind_ligand_codes = PDBbind_df.loc[:, "Lig_codes"]
    
    CASF2016_df = pd.read_csv("../data/affinity/CASF2016_data.tsv", sep = "\t")
    CASF2016_ligand_codes = CASF2016_df.loc[:, "Lig_codes"]
    
    CASF2013_df = pd.read_csv("../data/affinity/CASF2013_data.tsv", sep = "\t")
    CASF2013_ligand_codes = CASF2013_df.loc[:, "Lig_codes"]
    
    CSAR2014_df = pd.read_csv("../data/affinity/CSAR2014_data.tsv", sep = "\t")
    CSAR2014_ligand_codes = CSAR2014_df.loc[:, "Lig_codes"]
    
    CSAR2012_df = pd.read_csv("../data/affinity/CSAR2012_data.tsv", sep = "\t")
    CSAR2012_ligand_codes = CSAR2012_df.loc[:, "Lig_codes"]
    
    CSARset1_df = pd.read_csv("../data/affinity/CSARset1_data.tsv", sep = "\t")
    CSARset1_ligand_codes = CSARset1_df.loc[:, "Lig_codes"]
    
    CSARset2_df = pd.read_csv("../data/affinity/CSARset2_data.tsv", sep = "\t")
    CSARset2_ligand_codes = CSARset2_df.loc[:, "Lig_codes"]
    
    Astex_df = pd.read_csv("../data/affinity/Astex_data.tsv", sep = "\t")
    Astex_ligand_codes = Astex_df.loc[:, "Lig_codes"]
    
    total_lig_codes = np.unique(np.concatenate((training_lig_codes, CASF2016_lig_codes, CASF2013_lig_codes, CSAR2014_lig_codes, CSAR2012_lig_codes, CSARset1_lig_codes, CSARset2_lig_codes, Astex_lig_codes)))
    print(f"[Binding affinity datasets] total unique compounds: {len(total_lig_codes)}")
    print()
    
    # Get compound features
    atom_slices, edge_slices = [0], [0]
    all_atom_features, all_edge_features = list(), list()
    edge_indices, total_n_atoms, id_list = list(), list(), list()

    total_atoms, total_edges = 0, 0
    avg_degree = 0
    
    path = "../data/compound_raw_graph/"
    for mol_idx, lig_code in tqdm(enumerate(total_lig_codes)):
        if mol_idx % 500 == 0:
            print(mol_idx, len(total_lig_codes))
        sdf_path = os.path.join(path, f"{lig_code}_ideal.sdf")
        suppl = Chem.SDMolSupplier(sdf_path, removeHs = True)

        mol = next(iter(suppl))

        n_atoms = len(mol.GetAtoms())
        
        atom_features_list, edge_index, edge_features, n_edges = get_mol_features(mol)

        if lig_code in ["313", "M2T"]:
            n_atoms, atom_features_list, edge_features, edge_index, n_edges = remove_hydrogen(atom_features_list, edge_index, edge_features)

        all_atom_features.append(torch.tensor(atom_features_list, dtype=torch.long))

        if atom_features_list == 0:
            print(sdf_path)
            return 

        avg_degree += (n_edges / 2) / n_atoms 
        edge_indices.append(edge_index)
        all_edge_features.append(edge_features)
        
        total_edges += n_edges
        total_atoms += n_atoms
        total_n_atoms.append(n_atoms)
        
        edge_slices.append(total_edges)
        atom_slices.append(total_atoms)
        
        id_list.append(lig_code)

    data_dict = {'mol_ids':id_list,
                 'n_atoms':torch.tensor(total_n_atoms, dtype=torch.long),
                 'atom_slices':torch.tensor(atom_slices, dtype=torch.long),
                 'edge_slices':torch.tensor(edge_slices, dtype=torch.long),
                 'edge_indices':torch.cat(edge_indices, dim=1),
                 'atom_features':torch.cat(all_atom_features, dim=0),
                 'edge_features':torch.cat(all_edge_features, dim=0),
                 'avg_degree':avg_degree / len(id_list)}

    torch.save(data_dict, f"../data/affinity/graph_data.pt")

    #################################
    # 4. COACH420 and HOLO4K datasets
    #################################
    COACH420_df = pd.read_csv("../data/pre-training/protein/COACH420_IS_data.tsv", sep = "\t")
    COACH420_ligand_codes = COACH420_df.loc[:, "LigandCodes"]
    
    HOLO4K_df = pd.read_csv("../data/pre-training/protein/HOLO4K_IS_data.tsv", sep = "\t")
    HOLO4K_ligand_codes = HOLO4K_df.loc[:, "LigandCodes"]
    
    total_lig_codes = np.unique(np.concatenate((COACH420_ligand_codes, HOLO4K_ligand_codes)))
    print(f"[COACH and HOLO4K datasets] total unique compounds: {len(total_lig_codes)}")
    print()
    
    # Get compound features
    atom_slices, edge_slices = [0], [0]
    all_atom_features, all_edge_features = list(), list()
    edge_indices, total_n_atoms, id_list = list(), list(), list()

    total_atoms, total_edges = 0, 0
    avg_degree = 0
    
    path = "../data/compound_raw_graph/"
    for mol_idx, lig_code in tqdm(enumerate(total_lig_codes)):
        if mol_idx % 500 == 0:
            print(mol_idx, len(total_lig_codes))
        sdf_path = os.path.join(path, f"{lig_code}_ideal.sdf")
        suppl = Chem.SDMolSupplier(sdf_path, removeHs = True)

        mol = next(iter(suppl))

        n_atoms = len(mol.GetAtoms())
        
        atom_features_list, edge_index, edge_features, n_edges = get_mol_features(mol)

        if lig_code in ["313", "M2T"]:
            n_atoms, atom_features_list, edge_features, edge_index, n_edges = remove_hydrogen(atom_features_list, edge_index, edge_features)

        all_atom_features.append(torch.tensor(atom_features_list, dtype=torch.long))

        if atom_features_list == 0:
            print(sdf_path)
            return 

        avg_degree += (n_edges / 2) / n_atoms 
        edge_indices.append(edge_index)
        all_edge_features.append(edge_features)
        
        total_edges += n_edges
        total_atoms += n_atoms
        total_n_atoms.append(n_atoms)
        
        edge_slices.append(total_edges)
        atom_slices.append(total_atoms)
        
        id_list.append(lig_code)

    data_dict = {'mol_ids':id_list,
                 'n_atoms':torch.tensor(total_n_atoms, dtype=torch.long),
                 'atom_slices':torch.tensor(atom_slices, dtype=torch.long),
                 'edge_slices':torch.tensor(edge_slices, dtype=torch.long),
                 'edge_indices':torch.cat(edge_indices, dim=1),
                 'atom_features':torch.cat(all_atom_features, dim=0),
                 'edge_features':torch.cat(all_edge_features, dim=0),
                 'avg_degree':avg_degree / len(id_list)}

    torch.save(data_dict, f"../data/protein/graph_data.pt")
    
if __name__ == "__main__":
    main()