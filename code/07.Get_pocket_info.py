import os
import sys
import yaml
import pickle
import random
import logging
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Subset, DataLoader

from common.utils import load_cfg
from protein_modules.trainer import Pseq2SitesTrainer
from protein_modules.loader import ProteinSequenceDataset


def main():
    
    #######################
    # 1. Load configuration
    #######################
    conf_path = "Pocket_predictor_configuration.yml"
    config = load_cfg(conf_path)
     
    device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['Train']['seed'])
     
    ###############
    # 2. Load data
    ###############
    BA_training_ID, BA_training_features, BA_training_BS, BA_training_seqs, BA_training_lengths = load_data(config["Path"]["BA_training_data"], config["Path"]["BA_training_prot_feats"])
    CASF2016_ID, CASF2016_features, CASF2016_BS, CASF2016_seqs, CASF2016_lengths = load_data(config["Path"]["CASF2016_data"], config["Path"]["CASF2016_prot_feats"])
    CASF2013_ID, CASF2013_features, CASF2013_BS, CASF2013_seqs, CASF2013_lengths = load_data(config["Path"]["CASF2013_data"], config["Path"]["CASF2013_prot_feats"])
    CSAR2014_ID, CSAR2014_features, CSAR2014_BS, CSAR2014_seqs, CSAR2014_lengths = load_data(config["Path"]["CSAR2014_data"], config["Path"]["CSAR2014_prot_feats"])
    CSAR2012_ID, CSAR2012_features, CSAR2012_BS, CSAR2012_seqs, CSAR2012_lengths = load_data(config["Path"]["CSAR2012_data"], config["Path"]["CSAR2012_prot_feats"])
    CSARset1_ID, CSARset1_features, CSARset1_BS, CSARset1_seqs, CSARset1_lengths = load_data(config["Path"]["CSARset1_data"], config["Path"]["CSARset1_prot_feats"])
    CSARset2_ID, CSARset2_features, CSARset2_BS, CSARset2_seqs, CSARset2_lengths = load_data(config["Path"]["CSARset2_data"], config["Path"]["CSARset2_prot_feats"])
    Astex_ID, Astex_features, Astex_BS, Astex_seqs, Astex_lengths = load_data(config["Path"]["Astex_data"], config["Path"]["Astex_prot_feats"])
    COACH420_ID, COACH420_features, COACH420_BS, COACH420_seqs, COACH420_lengths = load_data(config["Path"]["COACH420_data"], config["Path"]["COACH420_prot_feats"])
    HOLO4K_ID, HOLO4K_features, HOLO4K_BS, HOLO4K_seqs, HOLO4K_lengths = load_data(config["Path"]["HOLO4K_data"], config["Path"]["HOLO4K_prot_feats"])

    ###############
    # 3. Run test
    ###############
    for idx in range(1, 2):
        trainer = Pseq2SitesTrainer(config, device)
        trainer.model.load_state_dict(torch.load(config['Path']['save_path'] + f"/CV{idx}/Pocket_predictor.pth"))

        # BA training
        BA_Training_Dataset = ProteinSequenceDataset(PID = BA_training_ID, Pseqs = BA_training_seqs, Pfeatures = BA_training_features, Labels = BA_training_BS["4A"])
        BA_Training_Loader = DataLoader(BA_Training_Dataset, batch_size=config['Train']['batch_size'], shuffle=False)
        _, BA_Training_predictions = trainer.test(BA_Training_Loader)

        results_dict = dict()
        for protein_id, bs_prediction, protein_lengths in zip(BA_training_ID, BA_Training_predictions, BA_training_lengths):
            pred_list = remove_over_lengths(bs_prediction, protein_lengths, 0.55)
            results_dict[protein_id] = np.array(pred_list)

        with open(f"../results/pre-training/protein/CV{idx}/BA_Training_pocket.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        # CASF2016
        CASF2016_Dataset = ProteinSequenceDataset(PID = CASF2016_ID, Pseqs = CASF2016_seqs, Pfeatures = CASF2016_features, Labels = CASF2016_BS["4A"])
        CASF2016_Loader = DataLoader(CASF2016_Dataset, batch_size=config['Train']['batch_size'], shuffle=False)
        _, CASF2016_predictions = trainer.test(CASF2016_Loader)

        results_dict = dict()
        for protein_id, bs_prediction, protein_lengths in zip(CASF2016_ID, CASF2016_predictions, CASF2016_lengths):
            pred_list = remove_over_lengths(bs_prediction, protein_lengths, 0.55)
            results_dict[protein_id] = np.array(pred_list)
  
        with open(f"../results/pre-training/protein/CV{idx}/CASF2016_pocket.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        # CASF2013
        CASF2013_Dataset = ProteinSequenceDataset(PID = CASF2013_ID, Pseqs = CASF2013_seqs, Pfeatures = CASF2013_features, Labels = CASF2013_BS["4A"])
        CASF2013_Loader = DataLoader(CASF2013_Dataset, batch_size=config['Train']['batch_size'], shuffle=False)
        _, CASF2013_predictions = trainer.test(CASF2013_Loader)

        results_dict = dict()
        for protein_id, bs_prediction, protein_lengths in zip(CASF2013_ID, CASF2013_predictions, CASF2013_lengths):
            pred_list = remove_over_lengths(bs_prediction, protein_lengths, 0.55)
            results_dict[protein_id] = np.array(pred_list)

        with open(f"../results/pre-training/protein/CV{idx}/CASF2013_pocket.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        # CSAR2014
        CSAR2014_Dataset = ProteinSequenceDataset(PID = CSAR2014_ID, Pseqs = CSAR2014_seqs, Pfeatures = CSAR2014_features, Labels = CSAR2014_BS["4A"])
        CSAR2014_Loader = DataLoader(CSAR2014_Dataset, batch_size=config['Train']['batch_size'], shuffle=False)
        _, CSAR2014_predictions = trainer.test(CSAR2014_Loader)

        results_dict = dict()
        for protein_id, bs_prediction, protein_lengths in zip(CSAR2014_ID, CSAR2014_predictions, CSAR2014_lengths):
            pred_list = remove_over_lengths(bs_prediction, protein_lengths, 0.55)
            results_dict[protein_id] = np.array(pred_list)
            
        with open(f"../results/pre-training/protein/CV{idx}/CSAR2014_pocket.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        # CSAR2012
        CSAR2012_Dataset = ProteinSequenceDataset(PID = CSAR2012_ID, Pseqs = CSAR2012_seqs, Pfeatures = CSAR2012_features, Labels = CSAR2012_BS["4A"])
        CSAR2012_Loader = DataLoader(CSAR2012_Dataset, batch_size=config['Train']['batch_size'], shuffle=False)
        _, CSAR2012_predictions = trainer.test(CSAR2012_Loader)

        results_dict = dict()
        for protein_id, bs_prediction, protein_lengths in zip(CSAR2012_ID, CSAR2012_predictions, CSAR2012_lengths):
            pred_list = remove_over_lengths(bs_prediction, protein_lengths, 0.55)
            results_dict[protein_id] = np.array(pred_list)
  
        with open(f"../results/pre-training/protein/CV{idx}/CSAR2012_pocket.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        # CSARset1
        CSARset1_Dataset = ProteinSequenceDataset(PID = CSARset1_ID, Pseqs = CSARset1_seqs, Pfeatures = CSARset1_features, Labels = CSARset1_BS["4A"])
        CSARset1_Loader = DataLoader(CSARset1_Dataset, batch_size=config['Train']['batch_size'], shuffle=False)
        _, CSARset1_predictions = trainer.test(CSARset1_Loader)

        results_dict = dict()
        for protein_id, bs_prediction, protein_lengths in zip(CSARset1_ID, CSARset1_predictions, CSARset1_lengths):
            pred_list = remove_over_lengths(bs_prediction, protein_lengths, 0.55)
            results_dict[protein_id] = np.array(pred_list)
 
        with open(f"../results/pre-training/protein/CV{idx}/CSARset1_pocket.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        # CSARset2
        CSARset2_Dataset = ProteinSequenceDataset(PID = CSARset2_ID, Pseqs = CSARset2_seqs, Pfeatures = CSARset2_features, Labels = CSARset2_BS["4A"])
        CSARset2_Loader = DataLoader(CSARset2_Dataset, batch_size=config['Train']['batch_size'], shuffle=False)
        _, CSARset2_predictions = trainer.test(CSARset2_Loader)

        results_dict = dict()
        for protein_id, bs_prediction, protein_lengths in zip(CSARset2_ID, CSARset2_predictions, CSARset2_lengths):
            pred_list = remove_over_lengths(bs_prediction, protein_lengths, 0.55)
            results_dict[protein_id] = np.array(pred_list)
  
        with open(f"../results/pre-training/protein/CV{idx}/CSARset2_pocket.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        # Astex
        Astex_Dataset = ProteinSequenceDataset(PID = Astex_ID, Pseqs = Astex_seqs, Pfeatures = Astex_features, Labels = Astex_BS["4A"])
        Astex_Loader = DataLoader(Astex_Dataset, batch_size=config['Train']['batch_size'], shuffle=False)
        _, Astex_predictions = trainer.test(Astex_Loader)

        results_dict = dict()
        for protein_id, bs_prediction, protein_lengths in zip(Astex_ID, Astex_predictions, Astex_lengths):
            pred_list = remove_over_lengths(bs_prediction, protein_lengths, 0.55)
            results_dict[protein_id] = np.array(pred_list)

        with open(f"../results/pre-training/protein/CV{idx}/Astex_pocket.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        # COACH420
        COACH420_Dataset = ProteinSequenceDataset(PID = COACH420_ID, Pseqs = COACH420_seqs, Pfeatures = COACH420_features, Labels = COACH420_BS["4A"])
        COACH420_Loader = DataLoader(COACH420_Dataset, batch_size=config['Train']['batch_size'], shuffle=False)
        _, COACH420_predictions = trainer.test(COACH420_Loader)

        results_dict = dict()
        for protein_id, bs_prediction, protein_lengths in zip(COACH420_ID, COACH420_predictions, COACH420_lengths):
            pred_list = remove_over_lengths(bs_prediction, protein_lengths, 0.55)
            results_dict[protein_id] = np.array(pred_list)
 
        with open(f"../results/pre-training/protein/CV{idx}/COACH420_pocket.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        # HOLO4K
        HOLO4K_Dataset = ProteinSequenceDataset(PID = HOLO4K_ID, Pseqs = HOLO4K_seqs, Pfeatures = HOLO4K_features, Labels = HOLO4K_BS["4A"])
        HOLO4K_Loader = DataLoader(HOLO4K_Dataset, batch_size=config['Train']['batch_size'], shuffle=False)
        _, HOLO4K_predictions = trainer.test(HOLO4K_Loader)

        results_dict = dict()
        for protein_id, bs_prediction, protein_lengths in zip(HOLO4K_ID, HOLO4K_predictions, HOLO4K_lengths):
            pred_list = remove_over_lengths(bs_prediction, protein_lengths, 0.55)
            results_dict[protein_id] = np.array(pred_list)
  
        with open(f"../results/pre-training/protein/CV{idx}/HOLO4K_pocket.pkl", "wb") as f:
            pickle.dump(results_dict, f)

def load_data(label_path, features_path, train = False):
    with open(f"{features_path}", "rb") as f:
        features = pickle.load(f)
        
    df = pd.read_csv(f"{label_path}", sep = "\t")
    
    id_list = df.iloc[:, 0].values
    seqs_list = df.iloc[:, 1].values

    bs4A, bs8A = df.iloc[:,2].values, df.iloc[:,3].values
    bs_dict = {"4A":bs4A, "8A":bs8A}
    
    seqs_lengths = np.array([len(i) for i in seqs_list])
    
    return id_list, features, bs_dict, seqs_list, seqs_lengths
    
def remove_over_lengths(pred_list, length, thr):

    pred = list()
    
    for idx, val in enumerate(pred_list):
        if idx < length:

            if val > thr:
                pred.append(idx)

    return pred

if __name__  == "__main__":
    main()