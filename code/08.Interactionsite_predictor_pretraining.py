import os
import sys
import yaml
import math
import copy
import pickle
import random
import logging
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from common.utils import load_cfg

from interaction_modules.loaders import TrainDataset, pad_data
from interaction_modules.models import PretrainedBA
from interaction_modules.trainers import PCITrainer
import pandas as pd

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
    print("Pre-training interaction sites predictor...")
    conf_path = "Interaction_sites_predictor_configuration.yml"
    config = load_cfg(conf_path)
    device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu")
    
    #######################
    # 2. Load training data
    #######################
    training_data = pd.read_csv(f"{config['Path']['training_data']}", sep = "\t")
    PDB_IDs, Uniprot_IDs, Lig_codes = training_data.iloc[:, 0].values, training_data.iloc[:, 1].values, training_data.iloc[:, 2].values
    Interaction_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(PDB_IDs, Uniprot_IDs, Lig_codes)])

    ################################
    # 3. Split data to train and val
    ################################
    for idx in range(3): 
        all_idx = [i for i in range(len(Interaction_IDs))]
        TrainIdx = random.sample(all_idx, int(len(all_idx) * 0.9))
        ValIdx = list(set(all_idx) - set(TrainIdx))  
        
        with open(f"{config['Path']['save_path']}/CV{idx}/indexes.pkl", "wb") as f:
            pickle.dump((TrainIdx, ValIdx), f)
    
    BA_labels = [-1 for i in range(len(Interaction_IDs))]
    
    Dataset = TrainDataset(interaction_IDs = Interaction_IDs, labels = BA_labels, device = device,
                                    protein_features_path = config['Path']['training_protein_feat'],
                                    pocket_indices_path = config['Path']['training_pocket_ind'],
                                    compound_features_path = config['Path']['training_ligand_graph'],
                                    interaction_sites_path = config['Path']['training_interaction'])
         
    #################
    # 4. Run training
    #################
    for idx in range(3): 
        with open(f"{config['Path']['save_path']}/CV{idx}/indexes.pkl", "rb") as f:
            TrainIdx, ValIdx = pickle.load(f)
    
        reset_seed(config['Train']['seed'])
        
        PreTrainedBA_Model = PretrainedBA(config, device).cuda()
        checkpoint = torch.load(config["Path"]["compound_encoder_path"])
        PreTrainedBA_Model.compound_encoder.load_state_dict(checkpoint)

        for parameter in PreTrainedBA_Model.compound_encoder.parameters():
            parameter.requires_grad = False
        PreTrainedBA_Model.compound_encoder.eval()

        Parameters = [v for k, v in PreTrainedBA_Model.named_parameters() if (v.requires_grad == True) and ("ba_predictor" not in k)]
        Optimizer = optim.Adam([{'params':Parameters}], lr=config['Train']['lr'], weight_decay=config['Train']['decay'], amsgrad=False)        
        print('model trainable params: ', sum(p.numel() for p in PreTrainedBA_Model.parameters() if p.requires_grad))
        print('model Total params: ', sum(p.numel() for p in PreTrainedBA_Model.parameters()))
        
        TrainLoader = DataLoader(Subset(Dataset, TrainIdx), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        ValLoader = DataLoader(Subset(Dataset, ValIdx), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        print(f"> Training: {len(TrainIdx)}, Validation: {len(ValIdx)}")
        
        trainer = PCITrainer(config, PreTrainedBA_Model, Optimizer, device)

        for epoch in range(1, config["Train"]["epochs"] + 1):

            TrainLoss = trainer.PCITrain(TrainLoader)
            print(f"[Train ({epoch})] PairCE loss: {TrainLoss['InterSitesLoss']:.4f}")
        
            ValLoss, patience = trainer.PCIEval(ValLoader, idx)
            print(f"[Val ({epoch})] PairCE loss: {ValLoss['InterSitesLoss']:.4f}")

            if patience > config["Train"]["patience"]:
                print(f"Validation loss do not improves, stop training")
                break

if __name__ == "__main__":
    main()