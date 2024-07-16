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
    print("Pre-training pocket predictor...")
    conf_path = "Pocket_predictor_configuration.yml"
    config = load_cfg(conf_path)
    device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu")
    
    #######################
    # 2. Load training data
    #######################
    training_df = pd.read_csv(config["Path"]["training_data"], sep = "\t")
    IDs, seqs, prot_BS = training_df.iloc[:,0].values, training_df.iloc[:,1].values, training_df.iloc[:,3].values
    print(f"\t> Training seqs: {len(IDs)}")

    with open(config["Path"]["prot_feats"], "rb") as f:
        protein_feats = pickle.load(f)

    print(f"\t> Training features: {len(protein_feats)}")

    ################################
    # 3. Split data to train and val
    ################################
    for idx in range(1):
        all_idx = [i for i in range(len(IDs))]
        TrainIdx = random.sample(all_idx, int(len(all_idx) * 0.9))
        ValIdx = list(set(all_idx) - set(TrainIdx))  
        
        with open(f"{config['Path']['save_path']}/CV{idx}/index.pkl", "wb") as f:
            pickle.dump((TrainIdx, ValIdx), f)
           
    Dataset = ProteinSequenceDataset(PID = IDs, Pseqs = seqs, Pfeatures = protein_feats, Labels = prot_BS)
    
    #################
    # 4. Run training
    #################
    for idx in range(1):
        reset_seed(config['Train']['seed'])
        
        with open(f"{config['Path']['save_path']}/CV{idx}/index.pkl", "rb") as f:
            TrainIdx, ValIdx = pickle.load(f)
        
        TrainLoader = DataLoader(Subset(Dataset, TrainIdx), batch_size=config['Train']['batch_size'], shuffle=True)
        ValLoader = DataLoader(Subset(Dataset, ValIdx), batch_size=config['Train']['batch_size'], shuffle=True)

        trainer = Pseq2SitesTrainer(config, device)
        
        for epoch in range(1, config['Train']['epochs'] + 1):
            print(f"====Epoch: {epoch}====")
            train_loss = trainer.train(TrainLoader)
            print(f"[Train ({epoch})] Loss: {train_loss:.4f}")
            
            val_loss, patience = trainer.eval(ValLoader, idx)
            print(f"[Val ({epoch})] Loss: {val_loss:.4f}")

            if patience > config["Train"]["patience"]:
                print(f"Validation loss do not improves, stop training")
                break

if __name__ == "__main__":
    main()