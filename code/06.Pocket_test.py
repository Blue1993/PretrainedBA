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
    IDs, features, BS_labels, Seqs, Lengths = load_data(config["Path"]["training_data"], config["Path"]["prot_feats"])
    print(f"\t> Training seqs: {len(IDs)}")
    
    ###############
    # 3. Run test
    ###############
    '''
    Set a criterion threshold for binding sites based on G-mean and select the fold with the best performance.
    '''
    for idx in range(3):
        with open(f"{config['Path']['save_path']}/CV{idx}/indexes.pkl", "rb") as f:
            TrainIdx, ValIdx = pickle.load(f)
            TrainIdx = np.array(TrainIdx)
            ValIdx = np.array(ValIdx) 

        trainer = Pseq2SitesTrainer(config, device)
        trainer.model.load_state_dict(torch.load(config['Path']['save_path'] + f"/CV{idx}/Pocket_predictor.pth"))

        PSeqDataset = ProteinSequenceDataset(PID = IDs, Pseqs = Seqs, Pfeatures = features, Labels = BS_labels["8A"])
        
        PSeqTrainLoader = DataLoader(Subset(PSeqDataset, TrainIdx), batch_size=config['Train']['batch_size'], shuffle=False)
        PSeqValLoader = DataLoader(Subset(PSeqDataset, ValIdx), batch_size=config['Train']['batch_size'], shuffle=False)
        
        _, BSTrain_predictions = trainer.test(PSeqTrainLoader)
        _, BSVal_predictions = trainer.test(PSeqValLoader)
        
        print(f"CV{idx} Results ...")
        print("===============================================================")
        print("\t[Train] Performance comparision with difference thresholds")
        get_another_metric_results(IDs[TrainIdx], BS_labels['8A'][TrainIdx], BSTrain_predictions, Lengths[TrainIdx])
        print()
        print()
        
        print("\t[Validation] Performance comparision with difference thresholds")
        get_another_metric_results(IDs[ValIdx], BS_labels['8A'][ValIdx], BSVal_predictions, Lengths[ValIdx])
        print("===============================================================")
        print()
        print()

    ####################
    # 4. Get predictions
    ####################
    '''
    Predict binding sites based on the criterion threshold and fold set in step 3.
    '''
    for idx in range(1, 2):
        trainer = Pseq2SitesTrainer(config, device)
        trainer.model.load_state_dict(torch.load(config['Path']['save_path'] + f"/CV{idx}/Pocket_predictor.pth"))
        
        PSeqDataset = ProteinSequenceDataset(PID = IDs, Pseqs = Seqs, Pfeatures = features, Labels = BS_labels["8A"])
        PSeqLoader = DataLoader(PSeqDataset, batch_size=config['Train']['batch_size'], shuffle=False)

        _, BS_predictions = trainer.test(PSeqLoader)

        results_dict = dict()
        for protein_id, bs_prediction, protein_lengths in zip(IDs, BS_predictions, Lengths):
            pred_list = remove_over_lengths(bs_prediction, protein_lengths, 0.55)
            results_dict[protein_id] = np.array(pred_list)
            
        with open(f"../results/pre-training/protein/CV{idx}/BS_pocket.pkl", "wb") as f:
            pickle.dump(results_dict, f)


def load_data(label_path, features_path):
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

def get_another_metric_results(IDs, labels, predictions, lengths):
   
    train_resutls = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[],
                    12:[], 13:[], 14:[], 15:[], 16:[], 17:[]}
    convert = {0:0.1, 1:0.15, 2:0.2, 3:0.25, 4:0.30, 5:0.35, 6:0.40, 7:0.45, 8:0.50,
              9:0.55, 10:0.60, 11:0.65, 12:0.70, 13:0.75, 14:0.80, 15:0.85, 16:0.90, 17:0.95}

    for key in list(convert.keys()):
        thr = convert[key]

        total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0

        for idx, id_ in enumerate(IDs):
            total_indexes = [i for i in range(lengths[idx])]
            pocket_labels = list(map(int, labels[idx].split(",")))
            
            positive_predictions = remove_over_lengths(predictions[idx], lengths[idx], thr)

            negative_predictions = list(set(total_indexes) - set(positive_predictions))
            pocket_negative_labels = list(set(total_indexes) - set(pocket_labels))
            
            TP = len(set(positive_predictions) & set(pocket_labels))
            TN = len(set(negative_predictions) & set(pocket_negative_labels))
            FP = len(set(positive_predictions) & set(pocket_negative_labels))
            FN = len(set(negative_predictions) & set(pocket_labels)) 
            
            total_TP += TP
            total_TN += TN
            total_FP += FP
            total_FN += FN
        
        precision = total_TP / (total_TP + total_FP)
        recall = total_TP / (total_TP + total_FN)
        acc = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
        sensitivity = total_TP / (total_TP + total_FN)
        specificity = total_TN / (total_TN + total_FP) 
        gmean = np.sqrt(sensitivity * specificity)
        F2 = (5 * sensitivity * precision) / ((4*sensitivity) + precision)
        F1 = (2 * sensitivity * precision) / (sensitivity + precision)
        
        print(f"\t\t[Thr: {thr}] Precision: {precision:.4f}, Recall: {recall:.4f}, ACC: {acc:.4f}, Sensi: {sensitivity:.4f}, Spec: {specificity:.4f}, GMean: {gmean:.4f}, F2: {F2:.4f}, F1: {F1:.4f}\n")

if __name__  == "__main__":
    main()