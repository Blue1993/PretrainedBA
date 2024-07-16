import os
import torch 
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from sklearn import linear_model
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class PCITrainer():
    def __init__(self, config, model, optimizer, device):
        self.config = config
        self.model = model

        self.dta_criterion = nn.MSELoss()
        self.optimizer = optimizer 

        self.best_eval_loss = np.inf
        self.dta_best_eval_loss = np.inf

        self.patience = 0

    def PCITrain(self, Loader):        

        self.model.protein_encoder.train()
        self.model.cross_encoder.train()
        self.model.intersites_predictor.train()

        results = {"InterSitesLoss":0}  

        for idx, batch in enumerate(tqdm(Loader)):
            
            protein_features_set, compound_graphs, _, pocket_interaction_labels, _, _, pocket_mask, compound_mask = batch
            
            # forward
            _, pairwise_map_predictions, pairwise_mask = self.model(protein_features_set[0], compound_graphs, pocket_mask, compound_mask)
            
            # loss calculation
            pairwise_loss = self.pairwise_criterion(pairwise_map_predictions, pocket_interaction_labels, pairwise_mask)

            results['InterSitesLoss'] += float(pairwise_loss.cpu().item())
            
            self.optimizer.zero_grad()

            pairwise_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            
            self.optimizer.step()

        results = {k: v / len(Loader) for k, v in results.items()}

        return results

    @torch.no_grad()
    def PCIEval(self, Loader, fold):
        
        results = {"InterSitesLoss":0}  

        self.model.eval()
        
        for idx, batch in enumerate(tqdm(Loader)):
            with torch.no_grad():
                protein_features_set, compound_graphs, _, pocket_interaction_labels, _, _, pocket_mask, compound_mask = batch
                   
                # forward
                _, pairwise_map_predictions, pairwise_mask = self.model(protein_features_set[0], compound_graphs, pocket_mask, compound_mask)
                    
                # loss calculation
                pairwise_loss = self.pairwise_criterion(pairwise_map_predictions, pocket_interaction_labels, pairwise_mask)

                results['InterSitesLoss'] += float(pairwise_loss.cpu().item())

        results = {k: v / len(Loader) for k, v in results.items()}

        if results["InterSitesLoss"] < self.best_eval_loss:
            self.patience = 0
            
            torch.save(self.model.state_dict(), f"{self.config['Path']['save_path']}/CV{fold}/InteractionSite_predictor.pth")
            print(f"Save model improvements: {(self.best_eval_loss - results['InterSitesLoss']):.4f}")
            self.best_eval_loss = results['InterSitesLoss']
        else:
            self.patience += 1

        return results, self.patience
    
    def reconstruction_pairwise_map(self, pairwise_map_predictions, pocket_index_list, interaction_labels):
        final_pairwise_map = np.transpose(np.zeros(interaction_labels.shape), (0, 2, 1))
        pairwise_map_predictions = np.transpose(pairwise_map_predictions, (0, 2, 1))
        
        for idx, (pairwise_map, pocket_index) in enumerate(zip(pairwise_map_predictions, pocket_index_list)):
            for jdx, index in enumerate(pocket_index):
                if index != -1:
                    final_pairwise_map[idx, index, :] = pairwise_map[jdx]
                    
        return np.transpose(final_pairwise_map, (0, 2, 1))

    @torch.no_grad()
    def PCITest(self, Loader):
        
        pairwise_pred_list, pairwise_mask_list, pairwise_label_list, protein_lengths = list(), list(), list(), list()
        self.model.eval()
        
        for idx, batch in enumerate(tqdm(Loader)):
            with torch.no_grad():
                
                protein_features_set, compound_graphs, labels, pocket_interaction_labels, pairwise_map_labels, pairwise_mask, pocket_mask, compound_mask = batch
                    
                # forward
                ba_predictions, pairwise_map_predictions, pairwise_mask = self.model(protein_features_set[0], compound_graphs, pocket_mask, compound_mask)

                protein_lengths.extend(protein_features_set[2].detach().cpu().tolist())

                pairwise_map = self.reconstruction_pairwise_map(pairwise_map_predictions.detach().cpu().numpy(), protein_features_set[1].detach().cpu().numpy(), pairwise_map_labels.detach().cpu().numpy())

                for i in pairwise_map:
                    pairwise_pred_list.append(i)

                for i in pairwise_mask.detach().cpu().tolist():
                    pairwise_mask_list.append(i)
 
                for i in pairwise_map_labels.detach().cpu().tolist():
                    pairwise_label_list.append(i)

        return pairwise_pred_list, pairwise_mask_list, pairwise_label_list, protein_lengths
                
    def pairwise_criterion(self, pred, labels, mask, dim = None):

        loss_ft = nn.BCELoss(reduction = 'none')
        loss_all = loss_ft(pred, labels)

        loss = torch.sum(loss_all*mask) / pred.size()[0]

        return loss

class BATrainer():
    def __init__(self, config, model, optimizer, device):
        self.config = config
        self.model = model

        self.dta_criterion = nn.MSELoss()
        self.optimizer = optimizer

        self.best_eval_loss = np.inf
        self.patience = 0

    def DTATrain(self, Loader):        
        
        self.model.cross_encoder.train()
        self.model.intersites_predictor.train()
        self.model.ba_predictor.train() 

        pred_list, label_list = list(), list() 
        
        results = {"DTALoss":0, "InterSitesLoss":0}  

        for idx, batch in enumerate(tqdm(Loader)):
            
            protein_features_set, compound_graphs, labels, pocket_interaction_labels, _, _, pocket_mask, compound_mask = batch

            # forward
            ba_predictions, pairwise_map_predictions, pairwise_mask = self.model(protein_features_set[0], compound_graphs, pocket_mask, compound_mask)
            
            # loss calculation
            pairwise_loss = self.pairwise_criterion(pairwise_map_predictions, pocket_interaction_labels, pairwise_mask)
            ba_loss = self.dta_criterion(ba_predictions, labels)
            
            self.optimizer.zero_grad()
            loss = ba_loss + pairwise_loss * 0.8
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            
            self.optimizer.step()
            results['DTALoss'] += float(ba_loss.cpu().item())
            results['InterSitesLoss'] += float(pairwise_loss.cpu().item())

            pred_list.extend(ba_predictions.detach().cpu().tolist())
            label_list.extend(labels.detach().cpu().tolist())

        train_results = get_results(np.array(label_list), np.array(pred_list))
        results = {k: v / len(Loader) for k, v in results.items()}
        train_results = add_results(train_results)
        
        results.update(train_results)
        
        return results

    @torch.no_grad()
    def DTAEval(self, Loader, fold):
        
        results = {"DTALoss":0, "InterSitesLoss":0}  
                    
        pred_list, label_list = list(), list()
        self.model.eval()
        
        for idx, batch in enumerate(tqdm(Loader)):
            with torch.no_grad():
                protein_features_set, compound_graphs, labels, pocket_interaction_labels, _, _, pocket_mask, compound_mask = batch
                   
                # forward
                ba_predictions, pairwise_map_predictions, pairwise_mask = self.model(protein_features_set[0], compound_graphs, pocket_mask, compound_mask)
                    
                # loss calculation
                ba_loss = self.dta_criterion(ba_predictions, labels)
                pairwise_loss = self.pairwise_criterion(pairwise_map_predictions, pocket_interaction_labels, pairwise_mask)
                
                loss = ba_loss + pairwise_loss * 0.8

                results['DTALoss'] += float(ba_loss.cpu().item())
                results['InterSitesLoss'] += float(pairwise_loss.cpu().item())

                pred_list.extend(ba_predictions.detach().cpu().tolist())
                label_list.extend(labels.detach().cpu().tolist())
        
        eval_results = get_results(np.array(label_list), np.array(pred_list))
        eval_results = add_results(eval_results)
        results = {k: v / len(Loader) for k, v in results.items()}
        results.update(eval_results)

        if results["DTALoss"] < self.best_eval_loss:
            self.patience = 0
            
            torch.save(self.model.state_dict(), f"{self.config['Path']['save_path']}/CV{fold}/PretrainedBA.pth")
            print(f"Save model improvements: {(self.best_eval_loss - results['DTALoss']):.4f}")
            self.best_eval_loss = results['DTALoss']
        else:
            self.patience += 1

        return results, self.patience
    
    def reconstruction_pairwise_map(self, pairwise_map_predictions, pocket_index_list, interaction_labels):
        final_pairwise_map = np.transpose(np.zeros(interaction_labels.shape), (0, 2, 1))
        pairwise_map_predictions = np.transpose(pairwise_map, (0, 2, 1))
        
        for idx, (pairwise_map, pocket_index) in enumerate(zip(pairwise_map_predictions, pocket_index_list)):
            for jdx, index in enumerate(pocket_index):
                if index != -1:
                    final_pairwise_map[idx, index, :] = pairwise_map[jdx]
                    
        return np.transpose(final_pairwise_map, (0, 2, 1))

    @torch.no_grad()
    def DTATest(self, Loader):
        
        pred_list, label_list = list(), list()
         
        self.model.eval()
        
        for idx, batch in enumerate(tqdm(Loader)):
            with torch.no_grad():
                protein_features_set, compound_graphs, labels, pocket_interaction_labels, _, _, pocket_mask, compound_mask = batch
                    
                # forward
                ba_predictions, pairwise_map_predictions, pairwise_mask = self.model(protein_features_set[0], compound_graphs, pocket_mask, compound_mask)
                
                pred_list.extend(ba_predictions.detach().cpu().tolist())
                label_list.extend(labels.detach().cpu().tolist())  
        
        pred_list, label_list = np.array(pred_list), np.array(label_list)
        
        test_results = get_results(label_list, pred_list)
        results = add_results(test_results)
        
        return results, pred_list, label_list

    @torch.no_grad()
    def PairwiseMapTest(self, Loader):
        
        pairwise_pred_list, pairwise_mask_list, pairwise_label_list, protein_lengths = list(), list(), list(), list()
        self.model.eval()
        
        for idx, batch in enumerate(tqdm(Loader)):
            with torch.no_grad():
                
                protein_features_set, compound_graphs, labels, pocket_interaction_labels, pairwise_map_labels, pairwise_mask, pocket_mask, compound_mask = batch
                    
                # forward
                ba_predictions, pairwise_map_predictions, pairwise_mask = self.model(prot_feat_set[0], compound_graph, pmask, cmask)

                pairwise_map = self.return_matrix(pairwise_map.detach().cpu().numpy(), prot_feat_set[1].detach().cpu().numpy(), prot_feat_set[2].detach().cpu().numpy(), ori_interaction_labels.detach().cpu().numpy())
                
                protein_lengths.extend(prot_feat_set[2].detach().cpu().tolist())

                pairwise_map = self.reconstruction_pairwise_map(pairwise_map.detach().cpu().numpy(), prot_feat_set[1].detach().cpu().numpy(), ori_interaction_labels.detach().cpu().numpy())
                
                for i in pairwise_map:
                    pairwise_pred_list.append(i)

                for i in ori_pairwise_mask.detach().cpu().tolist():
                    pairwise_mask_list.append(i)
 
                for i in pairwise_map_labels.detach().cpu().tolist():
                    pairwise_label_list.append(i)

        return pairwise_pred_list, pairwise_mask_list, pairwise_label_list, protein_lengths
                
    def pairwise_criterion(self, pred, labels, mask, dim = None):

        loss_ft = nn.BCELoss(reduction = 'none')
        loss_all = loss_ft(pred, labels)

        loss = torch.sum(loss_all*mask) / pred.size()[0]

        return loss

def add_results(results):
    
    results_ = dict()
    
    for i, j in zip(["MSE", "MAE", "RMSE", "PCC", "SPEARMAN", "CI", "R2", "SD", "RM2"], results):
        results_[i] = j
    
    return results_

### Get results
def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)

def get_results(labels, predictions):
    
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    MSE = mean_squared_error(labels, predictions)
    RMSE = mean_squared_error(labels, predictions)**0.5
    MAE = mean_absolute_error(labels, predictions)
    PCC = pearsonr(labels, predictions)
    CI = concordance_index(labels, predictions)
    r2 = r2_score(labels, predictions)
    SPEARMAN = spearmanr(labels, predictions)
    
    regr = linear_model.LinearRegression()
    regr.fit(predictions.reshape(-1, 1), labels.reshape(-1, 1))
    testpredy = regr.predict(predictions.reshape(-1, 1))
    testmse = mean_squared_error(labels, testpredy.flatten())
    num = labels.shape[0]
    SD = np.sqrt((testmse * num) / (num -1))
    
    rm2 = get_rm2(labels, predictions)
    
    return MSE, MAE, RMSE, PCC[0], SPEARMAN[0], CI, r2, SD, rm2    
