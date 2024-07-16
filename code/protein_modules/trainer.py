import os
import pickle
import numpy as np
import transformers

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from protein_modules.models import Pseq2Sites

class Pseq2SitesTrainer():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Build model
        self.model = Pseq2Sites(self.config).to(self.device)
        self.best_eval_loss = np.inf
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr = 1e-3, weight_decay = 0.01)

    def train(self, loader):
        
        self.model.train()

        losses = 0.
        for batch in tqdm(loader):
            residue_data, prots_data, prots_mask, position_ids, labels = batch[0], batch[1], batch[2], batch[3], batch[4]

            # Forward
            _, _, BS_preds, _ = self.model(residue_data, prots_data, prots_mask, position_ids)
            
            # Cal loss
            self.optimizer.zero_grad() 
            loss = self.get_multi_label_loss(BS_preds, labels, prots_mask)
            
            loss.backward()
            
            # Backward
            self.optimizer.step()
            
            losses += float(loss.item())

        return losses / len(loader)
        
    def eval(self, loader, i):
        losses = 0.
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(loader):
            
                # Prepare input
                residue_data, prots_data, prots_mask, position_ids, labels = batch[0], batch[1], batch[2], batch[3], batch[4]
                
                # Forward 
                _, _, BS_preds, _ = self.model(residue_data, prots_data, prots_mask, position_ids)

                loss = self.get_multi_label_loss(BS_preds, labels, prots_mask)
                losses += loss.item()
        
        if self.best_eval_loss > losses / len(loader):
            torch.save(self.model.state_dict(), f"{self.config['Path']['save_path']}/CV{i}/Pocket_predictor.pth")
            print(f"Save model improvements: {(self.best_eval_loss - losses / len(loader)):.4f}")
            self.patience = 0
            self.best_eval_loss = losses / len(loader)
        else:
            self.patience += 1
            
        return losses / len(loader), self.patience

    def test(self, loader):
        losses, predictions = 0, list()
        self.model.eval()
        
        with torch.no_grad():
            #for batch in tqdm(loader):
            for batch in loader:
                residue_data, prots_data, prots_mask, position_ids, labels = batch[0], batch[1], batch[2], batch[3], batch[4]
                
                _, _, BS_preds, _ = self.model(residue_data, prots_data, prots_mask, position_ids)
                
                loss = self.get_multi_label_loss(BS_preds, labels, prots_mask)

                losses += loss.item()
                predictions.extend(torch.nn.functional.sigmoid(BS_preds).detach().tolist())
            
            return losses/ len(loader), np.array(predictions)
    '''
    def only_for_test(self, loader):
        predictions = list()
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(loader):
                prots_data, total_prots_data, prots_mask, position_ids = batch[0], batch[1], batch[2], batch[3]
                _, _, BS_preds, _ = self.model(prots_data, total_prots_data, prots_mask, position_ids)
                
                predictions.extend(torch.nn.functional.sigmoid(BS_preds).detach().tolist())
        
        return np.array(predictions)
    '''
    def get_multi_label_loss(self, predictions, labels, masks):
        
        weight = self.calculate_weights(labels, masks)
        loss_ft = nn.BCEWithLogitsLoss(weight = weight)
        loss = loss_ft(predictions, labels)  
        return loss
        
    def calculate_weights(self, labels, masks):
        labels_inverse = torch.abs(labels - torch.ones(labels.size()).cuda())
        
        negative_labels = labels_inverse * masks
        
        P = torch.sum(labels)
        N = torch.sum(negative_labels)

        P_weights = (P + N + 1) / (P + 1)
        N_weights = (P + N + 1) / (N + 1)

        weights = torch.multiply(labels, P_weights) + torch.multiply(negative_labels, N_weights)
        
        return weights 
        