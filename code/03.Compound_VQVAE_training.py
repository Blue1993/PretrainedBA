import os
import random
import pickle
import logging
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from common.utils import load_cfg
from compound_modules.pna import PNA
from compound_modules.trainers import VQVAETrainer
from compound_modules.lr_schedulers import WarmUpWrapper 
from compound_modules.loaders import MoleculeGraphDataset, graph_collate
from compound_modules.models import VectorQuantizer, GNNDecoder

NUM_NODE_ATTR = 119 
NUM_NODE_CHIRAL = 5
NUM_BOND_ATTR = 5

def main():
    
    #################
    # 1. Load config
    #################
    print("VQVAE training...")
    config_path = "VQVAE_configuration.yml"
    config = load_cfg(config_path)
    
    torch.manual_seed(config['Train']['seed'])
    np.random.seed(config['Train']['seed'])
    
    device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['Train']['seed'])
        
    #############################
    # 2. Set up Molecule dataset
    #############################
    print("Set up Molecule dataset ...")
    MGraphDataset = MoleculeGraphDataset(processed_file=config['Path']['dataset_path'])
    print(f"Dataset size: {len(MGraphDataset)}")
    
    all_idx = [i for i in range(len(MGraphDataset))]
    MGraphTrainIdx = random.sample(all_idx, int(len(all_idx) * 0.9))
    MGraphValIdx = list(set(all_idx) - set(MGraphTrainIdx))

    MGraphTrainLoader = DataLoader(Subset(MGraphDataset, MGraphTrainIdx), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=graph_collate)
    MGraphValLoader = DataLoader(Subset(MGraphDataset, MGraphValIdx), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=graph_collate)

    #################
    # 3. Define model
    #################
    MGraphModel = PNA(**config["MGraphModel"]["architecture"]).to(device)
    
    vq_layer = VectorQuantizer(embedding_dim=config['Architecture']['emb_dim'], num_embeddings=config['Architecture']['num_tokens'], commitment_cost=config['Architecture']['commitment_cost']).to(device)
    atom_pred_decoder = GNNDecoder(hidden_dim=config['Architecture']['emb_dim'], out_dim=NUM_NODE_ATTR, JK=config['Architecture']['JK'], gnn_type=config['Architecture']['gnn_type']).to(device)
    atom_chiral_pred_decoder = GNNDecoder(hidden_dim=config['Architecture']['emb_dim'], out_dim=NUM_NODE_CHIRAL, JK=config['Architecture']['JK'], gnn_type=config['Architecture']['gnn_type']).to(device)
    bond_pred_decoder = GNNDecoder(config['Architecture']['emb_dim'], NUM_BOND_ATTR, JK=config['Architecture']['JK'], gnn_type='linear').to(device)

    model_list = [MGraphModel, vq_layer, atom_pred_decoder, bond_pred_decoder, atom_chiral_pred_decoder]
      
    #####################
    # 4. Define optimizer 
    #####################
    optimizer_MGraph = optim.Adam(MGraphModel.parameters(), lr=config['Train']['lr'], weight_decay=config['Train']['decay'])
    optimizer_vq = optim.Adam(vq_layer.parameters(), lr=config['Train']['lr'], weight_decay=config['Train']['decay'])
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=config['Train']['lr'], weight_decay=config['Train']['decay'])
    optimizer_dec_pred_atoms_chiral = optim.Adam(atom_chiral_pred_decoder.parameters(), lr=config['Train']['lr'], weight_decay=config['Train']['decay'])
    optimizer_dec_pred_bonds = optim.Adam(bond_pred_decoder.parameters(), lr=config['Train']['lr'], weight_decay=config['Train']['decay'])
    
    optimizer_list = [optimizer_MGraph, optimizer_vq, optimizer_dec_pred_atoms, optimizer_dec_pred_atoms_chiral, optimizer_dec_pred_bonds]
    
    #####################
    # 5. Define scheduler
    #####################
    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / config['Train']['epochs']) ) * 0.5
    
    scheduler_vq = torch.optim.lr_scheduler.LambdaLR(optimizer_vq, lr_lambda=scheduler)
    scheduler_dec_atom = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms, lr_lambda=scheduler)
    scheduler_dec_chiral = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms_chiral, lr_lambda=scheduler)  
    scheduler_dec_bond = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_bonds, lr_lambda=scheduler)  

    scheduler_MGraph = WarmUpWrapper(optimizer_MGraph, **config["Scheduler"]["params"])
    
    ###################
    # 6. Define trainer
    ###################
    trainer = VQVAETrainer(config, model_list, optimizer_list, scheduler_MGraph, device)
    print('model trainable params: ', sum(p.numel() for p in MGraphModel.parameters() if p.requires_grad))
    
    ##############
    # 7. Run train
    ##############
    for epoch in range(1, config['Train']['epochs']+1):
        print("====epoch " + str(epoch) + " ====")
        train_loss = trainer.train(MGraphTrainLoader)
        print(f"[Train ({epoch})] Loss: {train_loss['loss_accum']:.4f}, VQ loss: {train_loss['vq_loss_accum']:.4f}," +  
                f"Atom loss: {train_loss['atom_loss_accum']:.4f}, Atom chiral loss: {train_loss['atom_chiral_loss_accum']:.4f}, Edge loss: {train_loss['edge_loss_accum']:.4f}")

        val_loss, patience = trainer.eval(MGraphValLoader)
        print(f"[Validation ({epoch})] Loss: {val_loss['loss_accum']:.4f}, VQ loss: {val_loss['vq_loss_accum']:.4f}," + 
                f"Atom loss: {val_loss['atom_loss_accum']:.4f}, Atom chiral loss: {val_loss['atom_chiral_loss_accum']:.4f}, Edge loss: {val_loss['edge_loss_accum']:.4f}")

        if scheduler_vq is not None:
            scheduler_vq.step()
        if scheduler_dec_atom is not None:
            scheduler_dec_atom.step()
        if scheduler_dec_chiral is not None:
            scheduler_dec_chiral.step()  
        if scheduler_dec_bond is not None:
            scheduler_dec_bond.step()

        if patience > config["Train"]["patience"]:
            break
    
if __name__ == "__main__":
    main()
