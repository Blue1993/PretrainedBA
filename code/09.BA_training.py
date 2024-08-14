import os
import sys
import yaml
import pickle
import random
import logging
import copy
import math
import pandas as pd
import numpy as np

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
    print("Run PDBbind training")
    conf_path = "BA_predictor_configuration.yml"
    
    config = load_cfg(conf_path)    
    device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu") 
    
    ##############
    # 2. Load Data
    ##############
    with open(config["Path"]["Kfold"], "rb") as f:
        Kfold_index_dict = pickle.load(f)

    Training_df = pd.read_csv(config["Path"]["training_data"], sep = "\t")
    print(f"[Training] number of complexes: {len(Training_df)}")
    PDB_IDs, Uniprot_IDs, Lig_codes, BA_labels = Training_df.iloc[:, 0].values, Training_df.iloc[:, 1].values, Training_df.iloc[:, 2].values, Training_df.iloc[:, 3].values
    Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(PDB_IDs, Uniprot_IDs, Lig_codes)]) 

    #########################
    # 3. CV setting and Train
    ######################### 
    for idx in range(5):
        reset_seed(2024)
        
        BADataset = TrainDataset(interaction_IDs = Interactions_IDs, labels = BA_labels, device = device,
                                        protein_features_path = config["Path"]["training_protein_feat"],
                                        pocket_indices_path = config["Path"]["training_pocket_ind"],
                                        compound_features_path = config["Path"]["training_ligand_graph"],
                                        interaction_sites_path = config["Path"]["training_interaction"],)
                                        
        print(f">>> CV {idx} is running ...")
        train_index, val_index, test_index = Kfold_index_dict[idx]["train"], Kfold_index_dict[idx]["val"], Kfold_index_dict[idx]["test"]

        ###################################
        # 3.1 Define Binding Affinity model
        ###################################
        PreTrainedBA_Model = PretrainedBA(config, device).cuda()
        checkpoint = torch.load(config["Path"]["pretrained_interaction_predictor"])
        state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('ba_predictor')}
        PreTrainedBA_Model.load_state_dict(state_dict, strict=False)

        for parameter in PreTrainedBA_Model.compound_encoder.parameters():
            parameter.requires_grad = False
        PreTrainedBA_Model.compound_encoder.eval()

        for parameter in PreTrainedBA_Model.protein_encoder.parameters():
            parameter.requires_grad = False
        PreTrainedBA_Model.protein_encoder.eval()

        ###################
        # 3.2 Define params
        ###################
        parameters1 = [v for k, v in PreTrainedBA_Model.named_parameters() if "cross_encoder" in k or "intersites_predictor.pairwise_compound" in k or "intersites_predictor.pairwise_protein" in k]
        parameters2 = [v for k, v in PreTrainedBA_Model.named_parameters() if "ba_predictor" in k or "intersites_predictor.latent_compound" in k or "intersites_predictor.latent_protein" in k]
        optimizer = optim.Adam([{'params':parameters1, "lr":1e-10}, {'params':parameters2, "lr":1e-10}], amsgrad=False)

        ########################
        # 3.3 Parms for training 
        ########################       
        scheduler_dta = CosineAnnealingWarmUpRestarts(optimizer, T_0=15, T_mult=1, eta_maxes=[5e-4, 1e-3], T_up=1, gamma=0.99) 
        print('model trainable params: ', sum(p.numel() for p in PreTrainedBA_Model.parameters() if p.requires_grad))
        print('model Total params: ', sum(p.numel() for p in PreTrainedBA_Model.parameters()))
        
        #######################
        # 3.4 Define dataloader 
        #######################
        train_sampler = torch.utils.data.RandomSampler(
            Subset(BADataset, train_index), 
            generator=torch.Generator().manual_seed(2024)
        )
        
        BATrainLoader = DataLoader(Subset(BADataset, train_index), batch_size=config['Train']['batch_size'], sampler=train_sampler, collate_fn=pad_data)
        BAValLoader = DataLoader(Subset(BADataset, val_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        print(f"> Training: {len(train_index)}, Validation: {len(val_index)}, Test: {len(test_index)}")

        trainer = BATrainer(config, PreTrainedBA_Model, optimizer, device)

        ##############
        # 3.5 BA Train
        ##############
        for epoch in range(1, config["Train"]["epochs"] + 1):

            TrainLoss = trainer.DTATrain(BATrainLoader)
            print(f"[Train ({epoch})] BA loss: {TrainLoss['DTALoss']:.4f}, PairCE loss: {TrainLoss['InterSitesLoss']:.4f}, PCC: {TrainLoss['PCC']:.4f}")
        
            ValLoss, patience = trainer.DTAEval(BAValLoader, idx)
            print(f"[Val ({epoch})] BA loss: {ValLoss['DTALoss']:.4f}, PairCE loss: {ValLoss['InterSitesLoss']:.4f}, PCC: {ValLoss['PCC']:.4f}")

            if patience > config["Train"]["patience"]:
                print(f"Validation loss do not improves, stop training")
                break
                
            if scheduler_dta is not None:
                scheduler_dta.step()                 
            print()
      
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_maxes=[0.1, 0.1], T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_maxes = eta_maxes
        self.eta_maxes = eta_maxes
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(eta_max - base_lr) * self.T_cur / self.T_up + base_lr 
                    for eta_max, base_lr in zip(self.eta_maxes, self.base_lrs)]
        else:
            return [base_lr + (eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for eta_max, base_lr in zip(self.eta_maxes, self.base_lrs)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_maxes = [base_eta_max * (self.gamma ** self.cycle) for base_eta_max in self.base_eta_maxes]
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

if __name__ == "__main__":
    main()