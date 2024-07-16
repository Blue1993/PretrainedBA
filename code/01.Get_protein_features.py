import re
import torch
import pickle
import argparse
import numpy as np
import pandas as pd

from protein_modules.Get_protein_features import get_protein_features
from transformers import BertModel, BertTokenizer

def main():

    #########################
    # 1. Pre-training dataset
    #########################
    Pretraining_df = pd.read_csv("../data/pre-training/interaction/interaction_sites_predictor_training_data.tsv", sep = "\t")
    Pretraining_IDs, Pretraining_seqs = Pretraining_df.UniprotIDs.values, Pretraining_df.UniprotSeqs.values
    
    protein_seqs_dict = dict()
    for i, s in zip(Pretraining_IDs, Pretraining_seqs):
        protein_seqs_dict[i] = s
    print(f"[Pre-training dataset] Uniprot_IDs: {len(protein_seqs_dict)}")
    
    protein_features_dict = get_protein_features(protein_seqs_dict)
    
    with open("../data/pre-training/protein/pocket_predictor_training_features.pkl", "wb") as f:        
        pickle.dump(protein_features_dict, f)
        
    ##############################
    # 2. Binding affinity datasets
    ##############################
    # PDBbind
    PDBbind_df = pd.read_csv("../data/affinity/training_data.tsv", sep = "\t")
    PDBbind_IDs, PDBbind_seqs = PDBbind_df.Uniprot_IDs.values, PDBbind_df.Uniprot_seqs.values
    
    protein_seqs_dict, protein_features_dict = dict(), dict()
    for i, s in zip(prots_IDs, prots_seqs):
        protein_seqs_dict[i] = s
    print(f"[PDBbind dataset] Uniprot_IDs: {len(protein_seqs_dict)}")
    
    protein_features_dict = get_protein_features(protein_seqs_dict)
    
    with open("../data/affinity/protein/training_protein_features.pkl", "wb") as f:        
        pickle.dump(protein_features_dict, f)
    
    # CASF2016
    CASF2016_df = pd.read_csv("../data/affinity/CASF2016_data.tsv", sep = "\t")
    CASF2016_IDs, CASF2016_seqs = CASF2016_df.Uniprot_IDs.values, CASF2016_df.Uniprot_seqs.values
    
    protein_seqs_dict = dict()
    for i, s in zip(CASF2016_IDs, CASF2016_seqs):
        protein_seqs_dict[i] = s
    print(f"[CASF2016 dataset] Uniprot_IDs: {len(protein_seqs_dict)}")
    
    protein_features_dict = get_protein_features(protein_seqs_dict)
    
    with open("../data/affinity/protein/CASF2016_protein_features.pkl", "wb") as f:        
        pickle.dump(protein_features_dict, f)
        
    # CASF2013
    CASF2013_df = pd.read_csv("../data/affinity/CASF2013_data.tsv", sep = "\t")
    CASF2013_IDs, CASF2013_seqs = CASF2013_df.Uniprot_IDs.values, CASF2013_df.Uniprot_seqs.values
    
    protein_seqs_dict = dict()
    for i, s in zip(CASF2013_IDs, CASF2013_seqs):
        protein_seqs_dict[i] = s
    print(f"[CASF2013 dataset] Uniprot_IDs: {len(protein_seqs_dict)}")
    
    protein_features_dict = get_protein_features(protein_seqs_dict)
    
    with open("../data/affinity/protein/CASF2013_protein_features.pkl", "wb") as f:        
        pickle.dump(protein_features_dict, f)
    
    # CSAR2014
    CSAR2014_df = pd.read_csv("../data/affinity/CSAR2014_data.tsv", sep = "\t")
    CSAR2014_IDs, CSAR2014_seqs = CSAR2014_df.Uniprot_IDs.values, CSAR2014_df.Uniprot_seqs.values
    
    protein_seqs_dict = dict()
    for i, s in zip(CSAR2014_IDs, CSAR2014_seqs):
        protein_seqs_dict[i] = s
    print(f"[CSAR2014 dataset] Uniprot_IDs: {len(protein_seqs_dict)}")
    
    protein_features_dict = get_protein_features(protein_seqs_dict)
    
    with open("../data/affinity/protein/CSAR2014_protein_features.pkl", "wb") as f:        
        pickle.dump(protein_features_dict, f)
    
    # CSAR2012
    CSAR2012_df = pd.read_csv("../data/affinity/CSAR2012_data.tsv", sep = "\t")
    CSAR2012_IDs, CSAR2012_seqs = CSAR2012_df.Uniprot_IDs.values, CSAR2012_df.Uniprot_seqs.values
    
    protein_seqs_dict = dict()
    for i, s in zip(CSAR2012_IDs, CSAR2012_seqs):
        protein_seqs_dict[i] = s
    print(f"[CSAR2012 dataset] Uniprot_IDs: {len(protein_seqs_dict)}")
    
    protein_features_dict = get_protein_features(protein_seqs_dict)
    
    with open("../data/affinity/protein/CSAR2012_protein_features.pkl", "wb") as f:        
        pickle.dump(protein_features_dict, f)
    
    # CSARset1
    CSARset1_df = pd.read_csv("../data/affinity/CSARset1_data.tsv", sep = "\t")
    CSARset1_IDs, CSARset1_seqs = CSARset1_df.Uniprot_IDs.values, CSARset1_df.Uniprot_seqs.values
    
    protein_seqs_dict = dict()
    for i, s in zip(CSARset1_IDs, CSARset1_seqs):
        protein_seqs_dict[i] = s
    print(f"[CSARset1 dataset] Uniprot_IDs: {len(protein_seqs_dict)}")
    
    protein_features_dict = get_protein_features(protein_seqs_dict)
    
    with open("../data/affinity/protein/CSARset1_protein_features.pkl", "wb") as f:        
        pickle.dump(protein_features_dict, f)
    
    # CSARset2
    CSARset2_df = pd.read_csv("../data/affinity/CSARset2_data.tsv", sep = "\t")
    CSARset2_IDs, CSARset2_seqs = CSARset2_df.Uniprot_IDs.values, CSARset2_df.Uniprot_seqs.values
    
    protein_seqs_dict = dict()
    for i, s in zip(CSARset2_IDs, CSARset2_seqs):
        protein_seqs_dict[i] = s
    print(f"[CSARset2 dataset] Uniprot_IDs: {len(protein_seqs_dict)}")
    
    protein_features_dict = get_protein_features(protein_seqs_dict)
    
    with open("../data/affinity/protein/CSARset2_protein_features.pkl", "wb") as f:        
        pickle.dump(protein_features_dict, f)
    
    # Astex
    Astex_df = pd.read_csv("../data/affinity/Astex_data.tsv", sep = "\t")
    Astex_IDs, Astex_seqs = Astex_df.Uniprot_IDs.values, Astex_df.Uniprot_seqs.values
    
    protein_seqs_dict = dict()
    for i, s in zip(Astex_IDs, Astex_seqs):
        protein_seqs_dict[i] = s
    print(f"[Astex dataset] Uniprot_IDs: {len(protein_seqs_dict)}")
    
    protein_features_dict = get_protein_features(protein_seqs_dict)
    
    with open("../data/affinity/protein/Astex_protein_features.pkl", "wb") as f:        
        pickle.dump(protein_features_dict, f)
    
    #################################
    # 3. COACH420 and HOLO4K datasets
    #################################
    COACH420_df = pd.read_csv("../data/pre-training/protein/COACH420_IS_data.tsv", sep = "\t")
    COACH420_IDs, COACH420_seqs = COACH420_df.Uniprot_IDs.values, COACH420_df.UniprotSeqs.values

    protein_seqs_dict = dict()
    for i, s in zip(COACH420_IDs, COACH420_seqs):
        protein_seqs_dict[i] = s
    print(f"[COACH420 dataset] Uniprot_IDs: {len(protein_seqs_dict)}")
    
    protein_features_dict = get_protein_features(protein_seqs_dict)
    
    with open("../data/pre-training/protein/COACH420_protein_features.pkl", "wb") as f:        
        pickle.dump(protein_features_dict, f)

    HOLO4K_df = pd.read_csv("../data/pre-training/protein/HOLO4K_IS_data.tsv", sep = "\t")
    HOLO4K_IDs, HOLO4K_seqs = HOLO4K_df.Uniprot_IDs.values, HOLO4K_df.UniprotSeqs.values
    
    protein_seqs_dict = dict()
    for i, s in zip(HOLO4K_IDs, HOLO4K_seqs):
        protein_seqs_dict[i] = s
    print(f"[HOLO4K dataset] Uniprot_IDs: {len(protein_seqs_dict)}")
    
    protein_features_dict = get_protein_features(protein_seqs_dict)
    
    with open("../data/pre-training/protein/HOLO4K_protein_features.pkl", "wb") as f:        
        pickle.dump(protein_features_dict, f)
        
if __name__ == "__main__":
    main()