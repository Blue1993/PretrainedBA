import os
import sys
import yaml
import pickle
import random
import logging
import argparse
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import _LRScheduler

from common.utils import load_cfg
from interaction_modules.trainers import BATrainer
from interaction_modules.models import PretrainedBA
from interaction_modules.loaders import TrainDataset, pad_data

def reset_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 
        
def main():

    ################
    # 1. Load config
    ################
    print("Test data")
    conf_path = "BA_predictor_configuration.yml"
    config = load_cfg(conf_path)
    device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu") 
    
    ##############
    # 2. Load data
    ##############
    with open(config["Path"]["Kfold"], "rb") as f:
        Kfold_index_dict = pickle.load(f)
        
    # For test
    Training_df = pd.read_csv(config["Path"]["training_data"], sep = "\t")
    print(f"[Training] number of complexes: {len(Training_df)}")
    Training_PDB_IDs, Training_Uniprot_IDs, Training_Lig_codes, Training_BA_labels = Training_df.iloc[:, 0].values, Training_df.iloc[:, 1].values, Training_df.iloc[:, 2].values, Training_df.iloc[:, 3].values
    Training_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(Training_PDB_IDs, Training_Uniprot_IDs, Training_Lig_codes)]) 
    
    BADataset = TrainDataset(interaction_IDs = Training_Interactions_IDs, labels = Training_BA_labels, device = device,
                                    protein_features_path = config["Path"]["training_protein_feat"],
                                    pocket_indices_path = config["Path"]["training_pocket_ind"],
                                    compound_features_path = config["Path"]["training_ligand_graph"],
                                    interaction_sites_path = config["Path"]["training_interaction"],)

    # CASF2016
    CASF2016_df = pd.read_csv(config["Path"]["CASF2016_data"], sep = "\t")
    print(f"[CASF2016] number of complexes: {len(CASF2016_df)}")
    CASF2016_PDB_IDs, CASF2016_Uniprot_IDs, CASF2016_Lig_codes, CASF2016_BA_labels = CASF2016_df.iloc[:, 0].values, CASF2016_df.iloc[:, 1].values, CASF2016_df.iloc[:, 2].values, CASF2016_df.iloc[:, 3].values
    CASF2016_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CASF2016_PDB_IDs, CASF2016_Uniprot_IDs, CASF2016_Lig_codes)])

    CASF2016_DTADataset = TrainDataset(interaction_IDs = CASF2016_Interactions_IDs, labels = CASF2016_BA_labels, device = device,
                                    protein_features_path = config["Path"]["CASF2016_protein_feat"],
                                    pocket_indices_path = config["Path"]["CASF2016_pocket_ind"],
                                    compound_features_path = config["Path"]["training_ligand_graph"],
                                    interaction_sites_path = config["Path"]["CASF2016_interaction"])
    CASF2016_Loader = DataLoader(CASF2016_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    # CASF2013
    CASF2013_df = pd.read_csv(config["Path"]["CASF2013_data"], sep = "\t")
    print(f"[CASF2013] number of complexes: {len(CASF2013_df)}")
    CASF2013_PDB_IDs, CASF2013_Uniprot_IDs, CASF2013_Lig_codes, CASF2013_BA_labels = CASF2013_df.iloc[:, 0].values, CASF2013_df.iloc[:, 1].values, CASF2013_df.iloc[:, 2].values, CASF2013_df.iloc[:, 3].values
    CASF2013_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CASF2013_PDB_IDs, CASF2013_Uniprot_IDs, CASF2013_Lig_codes)])

    CASF2013_DTADataset = TrainDataset(interaction_IDs = CASF2013_Interactions_IDs, labels = CASF2013_BA_labels, device = device,
                                    protein_features_path = config["Path"]["CASF2013_protein_feat"],
                                    pocket_indices_path = config["Path"]["CASF2013_pocket_ind"],
                                    compound_features_path = config["Path"]["training_ligand_graph"],
                                    interaction_sites_path = config["Path"]["CASF2013_interaction"])
    CASF2013_Loader = DataLoader(CASF2013_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    # CSAR2014
    CSAR2014_df = pd.read_csv(config["Path"]["CSAR2014_data"], sep = "\t")
    print(f"[CSAR2014] number of complexes: {len(CSAR2014_df)}")
    CSAR2014_PDB_IDs, CSAR2014_Uniprot_IDs, CSAR2014_Lig_codes, CSAR2014_BA_labels = CSAR2014_df.iloc[:, 0].values, CSAR2014_df.iloc[:, 1].values, CSAR2014_df.iloc[:, 2].values, CSAR2014_df.iloc[:, 3].values
    CSAR2014_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CSAR2014_PDB_IDs, CSAR2014_Uniprot_IDs, CSAR2014_Lig_codes)])

    CSAR2014_DTADataset = TrainDataset(interaction_IDs = CSAR2014_Interactions_IDs, labels = CSAR2014_BA_labels, device = device,
                                    protein_features_path = config["Path"]["CSAR2014_protein_feat"],
                                    pocket_indices_path = config["Path"]["CSAR2014_pocket_ind"],
                                    compound_features_path = config["Path"]["training_ligand_graph"],
                                    interaction_sites_path = config["Path"]["CSAR2014_interaction"])
    CSAR2014_Loader = DataLoader(CSAR2014_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    # CSAR2012
    CSAR2012_df = pd.read_csv(config["Path"]["CSAR2012_data"], sep = "\t")
    print(f"[CSAR2012] number of complexes: {len(CSAR2012_df)}")
    CSAR2012_PDB_IDs, CSAR2012_Uniprot_IDs, CSAR2012_Lig_codes, CSAR2012_BA_labels = CSAR2012_df.iloc[:, 0].values, CSAR2012_df.iloc[:, 1].values, CSAR2012_df.iloc[:, 2].values, CSAR2012_df.iloc[:, 3].values
    CSAR2012_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CSAR2012_PDB_IDs, CSAR2012_Uniprot_IDs, CSAR2012_Lig_codes)])

    CSAR2012_DTADataset = TrainDataset(interaction_IDs = CSAR2012_Interactions_IDs, labels = CSAR2012_BA_labels, device = device,
                                    protein_features_path = config["Path"]["CSAR2012_protein_feat"],
                                    pocket_indices_path = config["Path"]["CSAR2012_pocket_ind"],
                                    compound_features_path = config["Path"]["training_ligand_graph"],
                                    interaction_sites_path = config["Path"]["CSAR2012_interaction"])
    CSAR2012_Loader = DataLoader(CSAR2012_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    # CSARset1
    CSARset1_df = pd.read_csv(config["Path"]["CSARset1_data"], sep = "\t")
    print(f"[CSARset1] number of complexes: {len(CSARset1_df)}")
    CSARset1_PDB_IDs, CSARset1_Uniprot_IDs, CSARset1_Lig_codes, CSARset1_BA_labels = CSARset1_df.iloc[:, 0].values, CSARset1_df.iloc[:, 1].values, CSARset1_df.iloc[:, 2].values, CSARset1_df.iloc[:, 3].values
    CSARset1_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CSARset1_PDB_IDs, CSARset1_Uniprot_IDs, CSARset1_Lig_codes)])

    CSARset1_DTADataset = TrainDataset(interaction_IDs = CSARset1_Interactions_IDs, labels = CSARset1_BA_labels, device = device,
                                    protein_features_path = config["Path"]["CSARset1_protein_feat"],
                                    pocket_indices_path = config["Path"]["CSARset1_pocket_ind"],
                                    compound_features_path = config["Path"]["training_ligand_graph"],
                                    interaction_sites_path = config["Path"]["CSARset1_interaction"])
    CSARset1_Loader = DataLoader(CSARset1_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    # CSARset2
    CSARset2_df = pd.read_csv(config["Path"]["CSARset2_data"], sep = "\t")
    print(f"[CSARset2] number of complexes: {len(CSARset2_df)}")
    CSARset2_PDB_IDs, CSARset2_Uniprot_IDs, CSARset2_Lig_codes, CSARset2_BA_labels = CSARset2_df.iloc[:, 0].values, CSARset2_df.iloc[:, 1].values, CSARset2_df.iloc[:, 2].values, CSARset2_df.iloc[:, 3].values
    CSARset2_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CSARset2_PDB_IDs, CSARset2_Uniprot_IDs, CSARset2_Lig_codes)])

    CSARset2_DTADataset = TrainDataset(interaction_IDs = CSARset2_Interactions_IDs, labels = CSARset2_BA_labels, device = device,
                                    protein_features_path = config["Path"]["CSARset2_protein_feat"],
                                    pocket_indices_path = config["Path"]["CSARset2_pocket_ind"],
                                    compound_features_path = config["Path"]["training_ligand_graph"],
                                    interaction_sites_path = config["Path"]["CSARset2_interaction"])
    CSARset2_Loader = DataLoader(CSARset2_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    # Astex
    Astex_df = pd.read_csv(config["Path"]["Astex_data"], sep = "\t")
    print(f"[Astex] number of complexes: {len(Astex_df)}")
    Astex_PDB_IDs, Astex_Uniprot_IDs, Astex_Lig_codes, Astex_BA_labels = Astex_df.iloc[:, 0].values, Astex_df.iloc[:, 1].values, Astex_df.iloc[:, 2].values, Astex_df.iloc[:, 3].values
    Astex_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(Astex_PDB_IDs, Astex_Uniprot_IDs, Astex_Lig_codes)])

    Astex_DTADataset = TrainDataset(interaction_IDs = Astex_Interactions_IDs, labels = Astex_BA_labels, device = device,
                                    protein_features_path = config["Path"]["Astex_protein_feat"],
                                    pocket_indices_path = config["Path"]["Astex_pocket_ind"],
                                    compound_features_path = config["Path"]["training_ligand_graph"],
                                    interaction_sites_path = config["Path"]["Astex_interaction"])
    Astex_Loader = DataLoader(Astex_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
    print()
    
    ############
    # 3. BA Test
    ############           
    for idx in range(5):
        print(f"Test CV{idx}...")
        test_index = Kfold_index_dict[idx]["test"]
        
        PreTrainedBA_Model = PretrainedBA(config, device).cuda()
        checkpoint = torch.load(f"{config['Path']['save_path']}/CV{idx}/PretrainedBA.pth")
        PreTrainedBA_Model.load_state_dict(checkpoint)
        
        for parameter in PreTrainedBA_Model.parameters():
            parameter.requires_grad = False
        PreTrainedBA_Model.eval()
        
        trainer = BATrainer(config, PreTrainedBA_Model, None, device)
        
        # For test
        BALoader = DataLoader(Subset(BADataset, test_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        TestLoss, predictions, labels = trainer.DTATest(BALoader)
        print(f"[Test] RMSE: {TestLoss['RMSE']:.4f}, PCC: {TestLoss['PCC']:.4f}")
        fwrite(open(f"{config['Path']['test_results_path']}/CV{idx}/Test_results.txt", "w"), Training_Interactions_IDs[test_index], predictions, Training_BA_labels[test_index])
        
        # CASF2016
        TestLoss, predictions, labels = trainer.DTATest(CASF2016_Loader)
        print(f"[CASF2016] RMSE: {TestLoss['RMSE']:.4f}, PCC: {TestLoss['PCC']:.4f}")
        fwrite(open(f"{config['Path']['test_results_path']}/CV{idx}/CASF2016_results.txt", "w"), CASF2016_Interactions_IDs, predictions, CASF2016_BA_labels)
        
        # CASF2013
        TestLoss, predictions, labels = trainer.DTATest(CASF2013_Loader)
        print(f"[CASF2013] RMSE: {TestLoss['RMSE']:.4f}, PCC: {TestLoss['PCC']:.4f}")
        fwrite(open(f"{config['Path']['test_results_path']}/CV{idx}/CASF2013_results.txt", "w"), CASF2013_Interactions_IDs, predictions, CASF2013_BA_labels)
        
        # CSAR2014
        TestLoss, predictions, labels = trainer.DTATest(CSAR2014_Loader)
        print(f"[CSAR2014] RMSE: {TestLoss['RMSE']:.4f}, PCC: {TestLoss['PCC']:.4f}")
        fwrite(open(f"{config['Path']['test_results_path']}/CV{idx}/CSAR2014_results.txt", "w"), CSAR2014_Interactions_IDs, predictions, CSAR2014_BA_labels)
        
        # CSAR2012
        TestLoss, predictions, labels = trainer.DTATest(CSAR2012_Loader)
        print(f"[CSAR2012] RMSE: {TestLoss['RMSE']:.4f}, PCC: {TestLoss['PCC']:.4f}")
        fwrite(open(f"{config['Path']['test_results_path']}/CV{idx}/CSAR2012_results.txt", "w"), CSAR2012_Interactions_IDs, predictions, CSAR2012_BA_labels)

        # CSARset1
        TestLoss, predictions, labels = trainer.DTATest(CSARset1_Loader)
        print(f"[CSARset1] RMSE: {TestLoss['RMSE']:.4f}, PCC: {TestLoss['PCC']:.4f}")
        fwrite(open(f"{config['Path']['test_results_path']}/CV{idx}/CSARset1_results.txt", "w"), CSARset1_Interactions_IDs, predictions, CSARset1_BA_labels)
        
        # CSARset2
        TestLoss, predictions, labels = trainer.DTATest(CSARset2_Loader)
        print(f"[CSARset2] RMSE: {TestLoss['RMSE']:.4f}, PCC: {TestLoss['PCC']:.4f}")
        fwrite(open(f"{config['Path']['test_results_path']}/CV{idx}/CSARset2_results.txt", "w"), CSARset2_Interactions_IDs, predictions, CSARset2_BA_labels)

        # Astex
        TestLoss, predictions, labels = trainer.DTATest(Astex_Loader)
        print(f"[Astex] RMSE: {TestLoss['RMSE']:.4f}, PCC: {TestLoss['PCC']:.4f}")
        fwrite(open(f"{config['Path']['test_results_path']}/CV{idx}/Astex_results.txt", "w"), Astex_Interactions_IDs, predictions, Astex_BA_labels)
        print()
        
def fwrite(fw, interaction_ids, predictions, labels):
    fw.write(f"PDB_IDs\tLigand_Codes\tPredictions\tLabels\n")
    
    for ids, prediction, label in zip(interaction_ids, predictions, labels):
        fw.write(f"{ids.split('_')[0]}\t{ids.split('_')[2]}\t{prediction:.4f}\t{label:.4f}\n")
    
    fw.close()

if __name__ == "__main__":
    main()