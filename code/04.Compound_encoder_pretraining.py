import yaml
import pickle
import random
import logging
import numpy as np
from itertools import chain

import torch 
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from common.utils import load_cfg
from compound_modules.pna import PNA
from compound_modules.models import VectorQuantizer
from compound_modules.trainers import PreTrainCompound
from compound_modules.lr_schedulers import WarmUpWrapper
from compound_modules.loaders import MoleculeGraphDataset, contrastive_collate

def main():

    #################
    # 1. Load config
    #################
    print("Compound pretraining...")
    config_path = "Compound_encoder_configuration.yml"
    config = load_cfg(config_path)

    torch.manual_seed(config['MGraphTrain']['seed'])
    np.random.seed(config['MGraphTrain']['seed'])
    
    device = torch.device("cuda:" + str(config['MGraphTrain']['device'])) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['MGraphTrain']['seed'])

    ############################
    # 2. Set up Molecule dataset
    ############################
    print("Set up Molecule dataset ...")
    MGraphDataset = MoleculeGraphDataset(processed_file = config['MGraphPaths']['dataset_path'])

    all_idx = [i for i in range(len(MGraphDataset))]
    MGraphTrainIdx = random.sample(all_idx, int(len(all_idx) * 0.9))
    MGraphValIdx = list(set(all_idx) - set(MGraphTrainIdx))

    #######################
    # 3. Define Graph model
    #######################
    MGraphModel = PNA(**config["MGraphModel"]["architecture"]).to("cuda:0")
    
    model_parameters = {"embedding_dim":256, "num_embeddings":512, "commitment_cost":2.0}
    
    tokenizer = VectorQuantizer(**model_parameters).to(device)
    checkpoint = torch.load(f"../checkpoints/pre-training/compound/vqquantizer.pth")
    
    tokenizer.load_state_dict(checkpoint)
    for parameter in tokenizer.parameters():
        parameter.requires_grad = False
    tokenizer.eval()
    
    linear_pred_atoms1 = torch.nn.Linear(config["MGraphModel"]["architecture"]['hidden_dim'], 512).to(device)
    linear_pred_atoms2 = torch.nn.Linear(config["MGraphModel"]["architecture"]['hidden_dim'], 512).to(device)

    linear_pred_bonds1 = torch.nn.Linear(config["MGraphModel"]["architecture"]['hidden_dim'], 5).to(device)
    linear_pred_bonds2 = torch.nn.Linear(config["MGraphModel"]["architecture"]['hidden_dim'], 5).to(device)
    
    print('model trainable params: ', sum(p.numel() for p in MGraphModel.parameters() if p.requires_grad))

    #####################
    # 4. Define optimizer 
    #####################
    optimizer_linear_pred_atoms1 = optim.Adam(linear_pred_atoms1.parameters(), lr=float(config['MGraphTrain']['lr']), weight_decay=config['MGraphTrain']['decay'])
    optimizer_linear_pred_atoms2 = optim.Adam(linear_pred_atoms2.parameters(), lr=float(config['MGraphTrain']['lr']), weight_decay=config['MGraphTrain']['decay'])
    optimizer_linear_pred_bonds1 = optim.Adam(linear_pred_bonds1.parameters(), lr=float(config['MGraphTrain']['lr']), weight_decay=config['MGraphTrain']['decay'])
    optimizer_linear_pred_bonds2 = optim.Adam(linear_pred_bonds2.parameters(), lr=float(config['MGraphTrain']['lr']), weight_decay=config['MGraphTrain']['decay'])
    
    MGraph_optimizer = optim.Adam(MGraphModel.parameters(), lr=float(config['MGraphTrain']['lr']), weight_decay=config['MGraphTrain']['decay'])

    #####################
    # 5. Define scheduler
    #####################
    MGraphScheduler = WarmUpWrapper(MGraph_optimizer, **config["Scheduler"]["params"])
    
    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / config['MGraphTrain']['epochs']) ) * 0.5
    scheduler_pred_atoms1 = torch.optim.lr_scheduler.LambdaLR(optimizer_linear_pred_atoms1, lr_lambda=scheduler)
    scheduler_pred_atoms2 = torch.optim.lr_scheduler.LambdaLR(optimizer_linear_pred_atoms2, lr_lambda=scheduler)
    scheduler_pred_bonds1 = torch.optim.lr_scheduler.LambdaLR(optimizer_linear_pred_bonds1, lr_lambda=scheduler)
    scheduler_pred_bonds2 = torch.optim.lr_scheduler.LambdaLR(optimizer_linear_pred_bonds2, lr_lambda=scheduler)
    
    ###################
    # 6. Define trainer
    ###################
    model_list = [MGraphModel, linear_pred_atoms1, linear_pred_atoms2, linear_pred_bonds1, linear_pred_bonds2]
    optimizer_list = [MGraph_optimizer, optimizer_linear_pred_atoms1, optimizer_linear_pred_bonds1, optimizer_linear_pred_atoms2, optimizer_linear_pred_bonds2]
    trainer = PreTrainCompound(config=config, model_list=model_list, optimizer_list=optimizer_list, MGraphScheduler = MGraphScheduler, tokenizer = tokenizer, device=device)
    
    #################
    # 7. Run train
    #################
    for epoch in range(1, config['MGraphTrain']['epochs']+1):
        print(f"====Epoch: {epoch}====")
        MGraphTrainLoss = trainer.train(MGraphDataset, MGraphTrainIdx)
        print(f"[Train ({epoch})] CL: {MGraphTrainLoss['loss_cl']:.4f}, Tri: {MGraphTrainLoss['loss_tri']:.4f}, atom1: {MGraphTrainLoss['loss_atom_1']:.4f}, atom2: {MGraphTrainLoss['loss_atom_2']:.4f}, node acc: {MGraphTrainLoss['acc_node_accum']:.4f}, edge1: {MGraphTrainLoss['loss_edge_1']:.4f}, edge2: {MGraphTrainLoss['loss_edge_2']:.4f}, edge acc: {MGraphTrainLoss['acc_edge_accum']:.4f}, total: {MGraphTrainLoss['total_loss']:.4f}")

        MGraphValLoss, patience = trainer.eval(MGraphDataset, MGraphValIdx)
        print(f"[Val ({epoch})] CL: {MGraphValLoss['loss_cl']:.4f}, Tri: {MGraphValLoss['loss_tri']:.4f}, atom1: {MGraphValLoss['loss_atom_1']:.4f}, atom2: {MGraphValLoss['loss_atom_2']:.4f}, node acc: {MGraphValLoss['acc_node_accum']:.4f}, edge1: {MGraphValLoss['loss_edge_1']:.4f}, edge2: {MGraphValLoss['loss_edge_2']:.4f}, edge acc: {MGraphValLoss['acc_edge_accum']:.4f}, total: {MGraphValLoss['total_loss']:.4f}")

        if MGraphValLoss['acc_node_accum'] != 0.:
        
            if scheduler_pred_atoms1 is not None:
                scheduler_pred_atoms1.step()
            if scheduler_pred_atoms2 is not None:
                scheduler_pred_atoms2.step()
            if scheduler_pred_bonds1 is not None:
                scheduler_pred_bonds1.step()  
            if scheduler_pred_bonds2 is not None:
                scheduler_pred_bonds2.step()

        if patience > config["MGraphTrain"]["patience"]:
            break
            
if __name__ == "__main__":
    main()
    
    