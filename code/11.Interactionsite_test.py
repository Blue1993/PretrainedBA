import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
import numpy as np
import pandas as pd
from rdkit import Chem
import pickle
import os

import os
import sys
import yaml
import pickle
import random
import logging
import numpy as np
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from common.utils import load_cfg

from interaction_modules.loaders import TrainDataset, pad_data
from interaction_modules.models import PretrainedBA
from interaction_modules.trainers import PCITrainer
import pandas as pd
import math

def main():

    #################
    # 1. Load config
    #################
    conf_path = "Interaction_sites_predictor_configuration.yml"
    config = load_cfg(conf_path)    
    device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu")
    
    ###############
    # 2. Load data
    ###############
    COACH420_df = pd.read_csv(config["Path"]["COACH420_data"], sep = "\t")
    print(f"[COACH420] number of complexes: {len(COACH420_df)}")
    COACH420_PDB_IDs, COACH420_Uniprot_IDs, COACH420_Lig_codes = COACH420_df.iloc[:, 0].values, COACH420_df.iloc[:, 1].values, COACH420_df.iloc[:, 2].values
    COACH420_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(COACH420_PDB_IDs, COACH420_Uniprot_IDs, COACH420_Lig_codes)])
    COACH420_BA_labels = [-1 for i in range(len(COACH420_Interactions_IDs))]
    
    HOLO4K_df = pd.read_csv(config["Path"]["HOLO4K_data"], sep = "\t")
    print(f"[HOLO4K] number of complexes: {len(HOLO4K_df)}")
    HOLO4K_PDB_IDs, HOLO4K_Uniprot_IDs, HOLO4K_Lig_codes = HOLO4K_df.iloc[:, 0].values, HOLO4K_df.iloc[:, 1].values, HOLO4K_df.iloc[:, 2].values
    HOLO4K_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(HOLO4K_PDB_IDs, HOLO4K_Uniprot_IDs, HOLO4K_Lig_codes)])
    HOLO4K_BA_labels = [-1 for i in range(len(HOLO4K_Interactions_IDs))]

    #######################
    # 3. Define dataloader
    #######################
    COACH420_DTADataset = TrainDataset(interaction_IDs = COACH420_Interactions_IDs, labels = COACH420_BA_labels, device = device,
                                    protein_features_path = config["Path"]["COACH420_protein_feat"],
                                    pocket_indices_path = config["Path"]["COACH420_pocket_ind"],
                                    compound_features_path = config["Path"]["graph_path"],
                                    interaction_sites_path = config["Path"]["COACH420_interaction"])
    COACH420_Loader = DataLoader(COACH420_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)


    HOLO4K_DTADataset = TrainDataset(interaction_IDs = HOLO4K_Interactions_IDs, labels = HOLO4K_BA_labels, device = device,
                                    protein_features_path = config["Path"]["HOLO4K_protein_feat"],
                                    pocket_indices_path = config["Path"]["HOLO4K_pocket_ind"],
                                    compound_features_path = config["Path"]["graph_path"],
                                    interaction_sites_path = config["Path"]["HOLO4K_interaction"])
    HOLO4K_Loader = DataLoader(HOLO4K_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
    
    ####################
    # 4. Get predictions
    ####################
    for idx in range(5):
        PreTrainedBA_Model = PretrainedBA(config, device).cuda()
        checkpoint = torch.load(f"../checkpoints/affinity/CV{idx}/PretrainedBA.pth")
        PreTrainedBA_Model.load_state_dict(checkpoint)

        for parameter in PreTrainedBA_Model.parameters():
            parameter.requires_grad = False
        PreTrainedBA_Model.eval()
        
        trainer = PCITrainer(config, PreTrainedBA_Model, None, device)
        
        COACH420_pairwise_pre, COACH420_pairwise_mask, COACH420_pairwise_labels, COACH420_lengths = trainer.PCITest(COACH420_Loader)
        with open(f"../results/pre-training/interaction/CV{idx}/COACH420_interaction_score_results.pkl", "wb") as f:
            pickle.dump((COACH420_Interactions_IDs, COACH420_pairwise_pre, COACH420_lengths), f)
        
        HOLO4K_pairwise_pre, HOLO4K_pairwise_mask, HOLO4K_pairwise_labels, HOLO4K_lengths = trainer.PCITest(HOLO4K_Loader)
        with open(f"../results/pre-training/interaction/CV{idx}/HOLO4K_interaction_score_results.pkl", "wb") as f:
            pickle.dump((HOLO4K_Interactions_IDs, HOLO4K_pairwise_pre, HOLO4K_lengths), f)

    #################
    # 5. Get results
    #################
    """
    5.1 Atom-level results
    """
    for idx in range(5):
        print(f"Get atom-level results CV{idx}")

        # COACH420
        compound_results_df = get_results("Atom", COACH420_Interactions_IDs, config["Path"]["COACH420_interaction"], 
                f"../results/pre-training/interaction/CV{idx}/COACH420_interaction_score_results.pkl")
        print(f"[COACH420] AUPRC: {compound_results_df['AUPRC'].mean():.4f}")

        # HOLO4K
        compound_results_df = get_results("Atom", HOLO4K_Interactions_IDs, config["Path"]["HOLO4K_interaction"], 
                f"../results/pre-training/interaction/CV{idx}/HOLO4K_interaction_score_results.pkl")
        print(f"[HOLO4K] AUPRC: {compound_results_df['AUPRC'].mean():.4f}")
        print()
    print()
    
    """
    5.2 Residue-level results
    """
    for idx in range(5):
        print(f"Get Residue-level results CV{idx}")

        # COACH420
        protein_results_df = get_results("Residue", COACH420_Interactions_IDs, config["Path"]["COACH420_interaction"], 
            f"../results/pre-training/interaction/CV{idx}/COACH420_interaction_score_results.pkl")
        print(f"[COACH420] AUPRC: {protein_results_df['AUPRC'].mean():.4f}")
        
        # HOLO4K
        protein_results_df = get_results("Residue", HOLO4K_Interactions_IDs, config["Path"]["HOLO4K_interaction"], 
            f"../results/pre-training/interaction/CV{idx}/HOLO4K_interaction_score_results.pkl")
        print(f"[HOLO4K] AUPRC: {protein_results_df['AUPRC'].mean():.4f}")
        print()
    print()
    
    """
    5.3 Pair-level results
    """
    for idx in range(5):
        print(f"Get pair-level CV{idx}")

        # COACH420
        pair_results_df = get_results("Pair", COACH420_Interactions_IDs, config["Path"]["COACH420_interaction"], 
                f"../results/pre-training/interaction/CV{idx}/COACH420_interaction_score_results.pkl")
        print(f"[COACH420] AUPRC: {pair_results_df['AUPRC'].mean():.4f}")
        
        # HOLO4K
        pair_results_df = get_results("Pair", HOLO4K_Interactions_IDs, config["Path"]["HOLO4K_interaction"], 
                f"../results/pre-training/interaction/CV{idx}/HOLO4K_interaction_score_results.pkl")
        print(f"[HOLO4K] AUPRC: {pair_results_df['AUPRC'].mean():.4f}")
        print()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sorted_pred_labels(pred, labels, selected_count):
    results = [(p, l) for p, l in zip(pred, labels)]
    results.sort(key = lambda a: a[0], reverse = True)

    sorted_labels = np.array([i[1] for i in results])
    sorted_pred = np.zeros(sorted_labels.shape[0]).astype(np.int32)
    sorted_pred[:selected_count] = 1
    
    return sorted_pred, sorted_labels
    
def get_pair_level_metric(pairwise_pred, labels):

    num_vertex = labels.shape[0]
    num_residue = labels.shape[1]
    
    pairwise_pred = pairwise_pred[:num_vertex, :num_residue].reshape(-1)
    pairwise_label = labels.reshape(-1)

    precision, recall, thresholds = precision_recall_curve(pairwise_label, pairwise_pred)
    AUPRC = auc(recall, precision)

    return AUPRC, precision/(pos_ratio / (pos_ratio + neg_ratio))
    
def get_residue_level_metric(pairwise_pred, labels):

    num_vertex = labels.shape[0]
    num_residue = labels.shape[1]
    
    pairwise_pred = np.max(pairwise_pred[:num_vertex, :num_residue], axis = 0)
    pairwise_label = np.clip(np.sum(labels, axis = 0), 0, 1)

    precision, recall, thresholds = precision_recall_curve(pairwise_label, pairwise_pred)
    AUPRC = auc(recall, precision)

    return AUPRC, precision/(pos_ratio / (pos_ratio + neg_ratio))
    
def get_atom_level_metric(pairwise_pred, labels):

    num_vertex = labels.shape[0]
    num_residue = labels.shape[1]
    
    pairwise_pred = np.max(pairwise_pred[:num_vertex, :num_residue], axis = 1)
    pairwise_label = np.clip(np.sum(labels, axis = 1), 0, 1)

    try:
        AUROC = roc_auc_score(pairwise_label, pairwise_pred)

    except:
        return None, None

    precision, recall, thresholds = precision_recall_curve(pairwise_label, pairwise_pred)
    AUPRC = auc(recall, precision)

    return AUPRC
    
def get_results(method, interaction_ids, interaction_site_labels_path, interaction_site_predictions_path):
    
    data = {"PDBID":[], "LigID":[], "AUPRC":[]}
     
    with open(f"{interaction_site_predictions_path}", "rb") as f:
        interaction_site_predictions = pickle.load(f)
    
    with open(f"{interaction_site_labels_path}", "rb") as f:
        interaction_site_labels = pickle.load(f)

    for idx, sample_key in enumerate(interaction_site_predictions[0]):

        sample_key = f'{sample_key.split("_")[0]}_{sample_key.split("_")[2]}'
        sample_pred, sample_labels = interaction_site_predictions[1][idx], interaction_site_labels[sample_key][1]

        if method == "Atom":
            AUPRC = get_atom_level_metric(sample_pred, sample_labels)

        elif method == "Residue":
            AUPRC = get_residue_level_metric(sample_pred, sample_labels)

        elif method == "Pair":
            AUPRC = get_pair_level_metric(sample_pred, sample_labels)

        data["PDBID"].append(sample_key.split("_")[0])
        data["LigID"].append(sample_key.split("_")[1])
        data["AUPRC"].append(AUPRC)

    return pd.DataFrame(data)

if __name__ == "__main__":
    main()
