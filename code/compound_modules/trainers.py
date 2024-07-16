import dgl 
import copy
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, einsum
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Subset

import random
from compound_modules.lossess import NTXentMultiplePositives, NTXent
from compound_modules.loaders import DataLoaderMaskingPred
#from compound_modules.metrics import *

class VQVAETrainer():
    def __init__(self, config, model_list, optimizer_list, scheduler_MGraph, device):
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        
        self.MGraphModel = model_list[0]
        self.vq_layer = model_list[1]
        self.dec_pred_atoms = model_list[2]
        self.dec_pred_bonds = model_list[3]
        self.dec_pred_atoms_chiral = model_list[4]
        
        self.optimizer_MGraphModel = optimizer_list[0]
        self.optimizer_vq = optimizer_list[1]
        self.optimizer_dec_pred_atoms = optimizer_list[2]
        self.optimizer_dec_pred_bonds = optimizer_list[3]
        self.optimizer_dec_pred_atoms_chiral = optimizer_list[4]
        
        self.scheduler_MGraph = scheduler_MGraph
        
        self.best_eval_loss = np.inf 
        self.patience = 0
        
    def train(self, Loader):
        self.MGraphModel.train()
        self.vq_layer.train()
        self.dec_pred_atoms.train()
        self.dec_pred_bonds.train()
        self.dec_pred_atoms_chiral.train()
        
        total_resutls = {"loss_accum": 0, "vq_loss_accum": 0, "atom_loss_accum": 0, 
                            "atom_chiral_loss_accum": 0, "edge_loss_accum":0}
        
        for step, batch in enumerate(tqdm(Loader)):

            batch = move_to_device(batch, self.device)
            loss, loss_list, _, _ = self.process_batch(batch, optim = True)

            total_resutls["loss_accum"] += float(loss.cpu().item())
            total_resutls["vq_loss_accum"] += float(loss_list[0].cpu().item())
            total_resutls["atom_loss_accum"] += float(loss_list[1].cpu().item())
            total_resutls["atom_chiral_loss_accum"] += float(loss_list[2].cpu().item())
            total_resutls["edge_loss_accum"] += float(loss_list[3].cpu().item())

        return {k: total_resutls[k]/len(Loader) for k in list(total_resutls.keys())}
        
    def forward_pass(self, batch):
        ####################################################################################################################################   
        # atom features (9): [atomic_num, chirality, degree, formal_charge, numH, num_radical_e, hybridization, is_in_aromatic, is_in_ring]
        # edge features (4): [bond_type, bond_stereo, is_conjugated, bond_direction]
        ####################################################################################################################################

        atom_features = batch.ndata['feat'] # (B * num nodes, 9)
        edge_features = batch.edata['feat'] # (B * num edges, 4)

        edge_indices = torch.stack([batch.edges()[0], batch.edges()[1]], dim = 0) # (2, B * num edges) 
        node_representation, graph_representation = self.MGraphModel(batch)  # node repre: (B * num nodes, 256), graph repre: (B, 256)

        # VQVAE
        e, e_q_loss = self.vq_layer(atom_features, node_representation) # (B * num nodes, )

        # Atom
        pred_node = self.dec_pred_atoms(e, edge_indices, edge_features[:, :2]) # (B * num nodes, 119)
        pred_node_chiral = self.dec_pred_atoms_chiral(e, edge_indices, edge_features[:, :2]) # (B * num nodes, 5)

        atom_loss = self.criterion(pred_node, atom_features[:, 0])
        atom_chiral_loss = self.criterion(pred_node_chiral, atom_features[:, 1])
        recon_loss = atom_loss + atom_chiral_loss
        
        # Edge
        edge_rep = e[edge_indices[0]] + e[edge_indices[1]] # (B * num edges, 256)
        pred_edge = self.dec_pred_bonds(edge_rep, edge_indices, edge_features[:, :2]) # (B * num edges, 5)
        edge_loss = self.criterion(pred_edge, edge_features[:,0])
        recon_loss += edge_loss

        loss = recon_loss + e_q_loss
        
        return loss, (e_q_loss, atom_loss, atom_chiral_loss, edge_loss), node_representation, graph_representation

    def process_batch(self, batch, optim):

        loss, loss_list, node_representation, graph_representation = self.forward_pass(batch)
        
        if optim != False:
            loss.backward()
            
            self.optimizer_MGraphModel.step()
            self.optimizer_vq.step()
            self.optimizer_dec_pred_atoms.step()
            self.optimizer_dec_pred_bonds.step()
            self.optimizer_dec_pred_atoms_chiral.step()
            
            self.after_optim_step()
            
            self.optimizer_MGraphModel.zero_grad()
            self.optimizer_vq.zero_grad()
            self.optimizer_dec_pred_atoms.zero_grad()
            self.optimizer_dec_pred_bonds.zero_grad()
            self.optimizer_dec_pred_atoms_chiral.zero_grad()
        
        return loss, loss_list, node_representation, graph_representation
    
    def eval(self, Loader):
        self.MGraphModel.eval()
        self.vq_layer.eval()
        self.dec_pred_atoms.eval()
        self.dec_pred_bonds.eval()
        self.dec_pred_atoms_chiral.eval()
        
        total_resutls = {"loss_accum": 0, "vq_loss_accum": 0, "atom_loss_accum": 0, 
                            "atom_chiral_loss_accum": 0, "edge_loss_accum":0}
                            
        for step, batch in enumerate(tqdm(Loader)):

            batch = move_to_device(batch, self.device)
            loss, loss_list, _, _ = self.process_batch(batch, optim = False)
            
            total_resutls["loss_accum"] += float(loss.cpu().item())
            total_resutls["vq_loss_accum"] += float(loss_list[0].cpu().item())
            total_resutls["atom_loss_accum"] += float(loss_list[1].cpu().item())
            total_resutls["atom_chiral_loss_accum"] += float(loss_list[2].cpu().item())
            total_resutls["edge_loss_accum"] += float(loss_list[3].cpu().item())

        # Scheduler update
        self.step_schedulers(metrics=total_resutls["loss_accum"])

        total_resutls = {k: total_resutls[k]/len(Loader) for k in list(total_resutls.keys())}

        if self.best_eval_loss > total_resutls["loss_accum"]:

            torch.save(self.MGraphModel.state_dict(), f"{self.config['Path']['save_path']}/vqencoder.pth")
            torch.save(self.vq_layer.state_dict(), f"{self.config['Path']['save_path']}/vqquantizer.pth")

            print(f"Save model improvements: {(self.best_eval_loss - total_resutls['loss_accum']):.4f}")
            self.best_eval_loss = total_resutls["loss_accum"]
            self.patience = 0
            
        else:
            self.patience += 1
 
        return total_resutls, self.patience

    def after_optim_step(self):
        if self.scheduler_MGraph.total_warmup_steps > self.scheduler_MGraph._step:
            self.step_schedulers()

    def step_schedulers(self, metrics = None):
        try:
            self.scheduler_MGraph.step(metrics = metrics)
        except:
            self.scheduler_MGraph.step()
            
class PreTrainCompound():
    def __init__(self, config, model_list, optimizer_list, MGraphScheduler, tokenizer, device):
    #def __init__(self, config, model_list, optimizer_list, scheduler, device):    
        self.config = config
        self.device = device
        
        # MGraph
        self.MGraphModel = model_list[0]
        self.tokenizer = tokenizer
        
        self.linear_pred_atoms1 = model_list[1]
        self.linear_pred_atoms2 = model_list[2]
        self.linear_pred_bonds1 = model_list[3]
        self.linear_pred_bonds2 = model_list[4]
        
        self.MGraphOptimizer = optimizer_list[0]
        self.optimizer_linear_pred_atoms1 = optimizer_list[1]
        self.optimizer_linear_pred_atoms2 = optimizer_list[2]
        self.optimizer_linear_pred_bonds1 = optimizer_list[3]
        self.optimizer_linear_pred_bonds2 = optimizer_list[4]
        
        self.MGraphCriterion = nn.CrossEntropyLoss()
        self.MGraphScheduler = MGraphScheduler

        self.triplet_loss = nn.TripletMarginLoss(margin=0., p=2)
        
        self.best_eval_loss = np.inf
        self.patience = 0

    def train(self, MGraphDataset, MGraphIDX):
        
        MGraphResults = {"total_loss": 0, "loss_cl": 0, "loss_tri": 0, "loss_atom_1": 0, "loss_atom_2":0,
                                    "loss_edge_1": 0, "loss_edge_2": 0, "acc_node_accum": 0, "acc_edge_accum": 0}

        self.MGraphModel.train()
        self.linear_pred_atoms1.train()
        self.linear_pred_bonds1.train()
        self.linear_pred_atoms2.train()
        self.linear_pred_bonds2.train()

        random.shuffle(MGraphIDX)
        MGraphDataset1 = copy.deepcopy(MGraphDataset)
        MGraphDataset2 = copy.deepcopy(MGraphDataset)

        MGraphLoader1 = DataLoaderMaskingPred(Subset(MGraphDataset1, MGraphIDX), batch_size=self.config['MGraphTrain']['batch_size'], shuffle = False, mask_rate=self.config['MGraphTrain']['mask_rate1'], mask_edge=self.config['MGraphTrain']['mask_edge'])
        MGraphLoader2 = DataLoaderMaskingPred(Subset(MGraphDataset2, MGraphIDX), batch_size=self.config['MGraphTrain']['batch_size'], shuffle = False, mask_rate=self.config['MGraphTrain']['mask_rate2'], mask_edge=self.config['MGraphTrain']['mask_edge'])

        for MGraphStep, (batch1, batch2) in enumerate(zip(tqdm(MGraphLoader1), MGraphLoader2)):
            graph1, graph1_masked_atom_indices, graph1_masked_node_labels, graph1_masked_edge_indices, graph1_masked_edge_labels = move_to_device(batch1, self.device)
            graph2, graph2_masked_atom_indices, graph2_masked_node_labels, graph2_masked_edge_indices, graph2_masked_edge_labels = move_to_device(batch2, self.device)
 
            original_graph = copy.deepcopy(graph1)
            node_rep1, graph_rep1 = self.MGraphModel(graph1)  # node repre: (# of node, 200), graph repre: (# of samples, 256)
            node_rep2, graph_rep2 = self.MGraphModel(graph2)  # node repre: (# of node, 200), graph repre: (# of samples, 256)
                
            loss_cl = self.loss_cl(graph_rep1, graph_rep2)
            MGraphResults["loss_cl"] += float(loss_cl.cpu().item())

            original_graph.ndata['feat'][graph1_masked_atom_indices] = graph1_masked_node_labels
            original_graph.edata['feat'][graph1_masked_edge_indices] = graph1_masked_edge_labels
            original_graph.edata['feat'][graph1_masked_edge_indices + 1] = graph1_masked_edge_labels

            ##############
            #Get atom ids
            ##############
            original_graph_atom_rep, origin_graph_rep = self.MGraphModel(original_graph)  # node repre: (# of node, 200), graph repre: (# of samples, 256)

            with torch.no_grad(): 
                atom_ids = self.tokenizer.get_code_indices(original_graph.ndata['feat'], original_graph_atom_rep)
                labels1 = atom_ids[graph1_masked_atom_indices]
                labels2 = atom_ids[graph2_masked_atom_indices]

            loss_tri = self.loss_tri(origin_graph_rep, graph_rep1, graph_rep2)
            MGraphResults["loss_tri"] += float(loss_tri.cpu().item())

            loss_tricl = loss_cl + 0.1 * loss_tri

            pred_node1 = self.linear_pred_atoms1(node_rep1[graph1_masked_atom_indices])
            loss_mask_node1 = self.MGraphCriterion(pred_node1.double(), labels1)
            MGraphResults["loss_atom_1"] += float(loss_mask_node1.cpu().item())

            pred_node2 = self.linear_pred_atoms2(node_rep2[graph2_masked_atom_indices])
            loss_mask_node2 = self.MGraphCriterion(pred_node2.double(), labels2)
            MGraphResults["loss_atom_2"] += float(loss_mask_node2.cpu().item())
            
            acc_node1 = compute_accuracy(pred_node1, labels1)
            acc_node2 = compute_accuracy(pred_node2, labels2)
            
            acc_node = (acc_node1 + acc_node2) * 0.5
            MGraphResults["acc_node_accum"] += acc_node
            
            if self.config['MGraphTrain']['mask_edge']:
                masked_edge_index1_src, masked_edge_index1_des = graph1.edges()[0][graph1_masked_edge_indices], graph1.edges()[1][graph1_masked_edge_indices]
                edge_rep1 = node_rep1[masked_edge_index1_src] + node_rep1[masked_edge_index1_des]
                pred_edge1= self.linear_pred_bonds1(edge_rep1)
                
                loss_mask_edge1 = self.MGraphCriterion(pred_edge1.double(), graph1_masked_edge_labels[:,0])
                MGraphResults["loss_edge_1"] += float(loss_mask_edge1.cpu().item())
                
                masked_edge_index2_src, masked_edge_index2_des = graph2.edges()[0][graph2_masked_edge_indices], graph2.edges()[1][graph2_masked_edge_indices]
                edge_rep2 = node_rep2[masked_edge_index2_src] + node_rep2[masked_edge_index2_des]
                pred_edge2= self.linear_pred_bonds2(edge_rep2)
                loss_mask_edge2 = self.MGraphCriterion(pred_edge2.double(), graph2_masked_edge_labels[:,0])
                MGraphResults["loss_edge_2"] += float(loss_mask_edge2.cpu().item())
 
                acc_edge1 = compute_accuracy(pred_edge1, graph1_masked_edge_labels[:,0])
                acc_edge2 = compute_accuracy(pred_edge2, graph2_masked_edge_labels[:,0])
                acc_edge = (acc_edge1 + acc_edge2) * 0.5
                MGraphResults["acc_edge_accum"] += acc_edge

            loss = loss_tricl + loss_mask_node1 + loss_mask_node2 + loss_mask_edge1 + loss_mask_edge2
            MGraphResults["total_loss"] += float(loss.cpu().item())

            loss.backward()
            
            self.MGraphOptimizer.step()
            self.optimizer_linear_pred_atoms1.step()
            self.optimizer_linear_pred_bonds1.step()
            self.optimizer_linear_pred_atoms2.step()
            self.optimizer_linear_pred_bonds2.step()

            self.after_optim_step_MGraph()

            self.MGraphOptimizer.zero_grad()
            self.optimizer_linear_pred_atoms1.zero_grad()
            self.optimizer_linear_pred_bonds1.zero_grad()
            self.optimizer_linear_pred_atoms2.zero_grad()
            self.optimizer_linear_pred_bonds2.zero_grad()

        MGraphResults = {k: v / len(MGraphLoader1) for k, v in MGraphResults.items()}

        return MGraphResults
            
    def eval(self, MGraphDataset, MGraphIDX):
        MGraphResults = {"total_loss": 0, "loss_cl": 0, "loss_tri": 0, "loss_atom_1": 0, "loss_atom_2":0,
                                    "loss_edge_1": 0, "loss_edge_2": 0, "acc_node_accum": 0, "acc_edge_accum": 0}

        self.MGraphModel.eval()
        self.linear_pred_atoms1.eval()
        self.linear_pred_bonds1.eval()
        self.linear_pred_atoms2.eval()
        self.linear_pred_bonds2.eval()

        MGraphDataset1 = copy.deepcopy(MGraphDataset)
        MGraphDataset2 = copy.deepcopy(MGraphDataset)

        MGraphLoader1 = DataLoaderMaskingPred(Subset(MGraphDataset1, MGraphIDX), batch_size=self.config['MGraphTrain']['batch_size'], shuffle = False, mask_rate=self.config['MGraphTrain']['mask_rate1'], mask_edge=self.config['MGraphTrain']['mask_edge'])
        MGraphLoader2 = DataLoaderMaskingPred(Subset(MGraphDataset2, MGraphIDX), batch_size=self.config['MGraphTrain']['batch_size'], shuffle = False, mask_rate=self.config['MGraphTrain']['mask_rate2'], mask_edge=self.config['MGraphTrain']['mask_edge'])

        with torch.no_grad():

            #if self.iter_idx:
            for MGraphStep, (batch1, batch2) in enumerate(zip(tqdm(MGraphLoader1), MGraphLoader2)):
                graph1, graph1_masked_atom_indices, graph1_masked_node_labels, graph1_masked_edge_indices, graph1_masked_edge_labels = move_to_device(batch1, self.device)
                graph2, graph2_masked_atom_indices, graph2_masked_node_labels, graph2_masked_edge_indices, graph2_masked_edge_labels = move_to_device(batch2, self.device)
                
                original_graph = copy.deepcopy(graph1)
 
                node_rep1, graph_rep1 = self.MGraphModel(graph1)  # node repre: (# of node, 200), graph repre: (# of samples, 256)
                node_rep2, graph_rep2 = self.MGraphModel(graph2)  # node repre: (# of node, 200), graph repre: (# of samples, 256)

                loss_cl = self.loss_cl(graph_rep1, graph_rep2)
                MGraphResults["loss_cl"] += float(loss_cl.cpu().item())

                ### atom prediction ###
                #origin_graph = copy.deepcopy(graph1)
                original_graph.ndata['feat'][graph1_masked_atom_indices] = graph1_masked_node_labels
                original_graph.edata['feat'][graph1_masked_edge_indices] = graph1_masked_edge_labels
                original_graph.edata['feat'][graph1_masked_edge_indices + 1] = graph1_masked_edge_labels

                ##############
                #Get atom ids
                ##############
                original_graph_atom_rep, origin_graph_rep = self.MGraphModel(original_graph)  # node repre: (# of node, 200), graph repre: (# of samples, 256)
                atom_ids = self.tokenizer.get_code_indices(original_graph.ndata['feat'], original_graph_atom_rep)
                labels1 = atom_ids[graph1_masked_atom_indices]
                labels2 = atom_ids[graph2_masked_atom_indices]

                loss_tri = self.loss_tri(origin_graph_rep, graph_rep1, graph_rep2)
                MGraphResults["loss_tri"] += float(loss_tri.cpu().item())

                loss_tricl = loss_cl + 0.1 * loss_tri

                pred_node1 = self.linear_pred_atoms1(node_rep1[graph1_masked_atom_indices])
                loss_mask_node1 = self.MGraphCriterion(pred_node1.double(), labels1)
                MGraphResults["loss_atom_1"] += float(loss_mask_node1.cpu().item())

                pred_node2 = self.linear_pred_atoms2(node_rep2[graph2_masked_atom_indices])
                loss_mask_node2 = self.MGraphCriterion(pred_node2.double(), labels2)
                MGraphResults["loss_atom_2"] += float(loss_mask_node2.cpu().item())
                
                acc_node1 = compute_accuracy(pred_node1, labels1)
                acc_node2 = compute_accuracy(pred_node2, labels2)
                
                acc_node = (acc_node1 + acc_node2) * 0.5
                MGraphResults["acc_node_accum"] += acc_node
            
                if self.config['MGraphTrain']['mask_edge']:
                    masked_edge_index1_src, masked_edge_index1_des = graph1.edges()[0][graph1_masked_edge_indices], graph1.edges()[1][graph1_masked_edge_indices]
                    edge_rep1 = node_rep1[masked_edge_index1_src] + node_rep1[masked_edge_index1_des]
                    pred_edge1= self.linear_pred_bonds1(edge_rep1)
                    
                    loss_mask_edge1 = self.MGraphCriterion(pred_edge1.double(), graph1_masked_edge_labels[:,0])
                    MGraphResults["loss_edge_1"] += float(loss_mask_edge1.cpu().item())
                    
                    masked_edge_index2_src, masked_edge_index2_des = graph2.edges()[0][graph2_masked_edge_indices], graph2.edges()[1][graph2_masked_edge_indices]
                    edge_rep2 = node_rep2[masked_edge_index2_src] + node_rep2[masked_edge_index2_des]
                    pred_edge2= self.linear_pred_bonds2(edge_rep2)
                    loss_mask_edge2 = self.MGraphCriterion(pred_edge2.double(), graph2_masked_edge_labels[:,0])
                    MGraphResults["loss_edge_2"] += float(loss_mask_edge2.cpu().item())
     
                    acc_edge1 = compute_accuracy(pred_edge1, graph1_masked_edge_labels[:,0])
                    acc_edge2 = compute_accuracy(pred_edge2, graph2_masked_edge_labels[:,0])
                    acc_edge = (acc_edge1 + acc_edge2) * 0.5
                    MGraphResults["acc_edge_accum"] += acc_edge
                    
                loss = loss_tricl + loss_mask_node1 + loss_mask_node2 + loss_mask_edge1 + loss_mask_edge2
                MGraphResults["total_loss"] += float(loss.cpu().item())
            
            # Scheduler update
            self.step_graph_schedulers(metrics=MGraphResults['total_loss'])
            MGraphResults = {k: v / len(MGraphLoader1) for k, v in MGraphResults.items()}

        if MGraphResults['total_loss'] < self.best_eval_loss:
            self.patience = 0
            torch.save(self.MGraphModel.state_dict(), f"{self.config['MGraphPaths']['save_path']}/MGraphPretraingEncoder.pth")
            print(f"Save model improvements: {(self.best_eval_loss - MGraphResults['total_loss']):.4f}")
            self.best_eval_loss = MGraphResults['total_loss']
        else:
            self.patience += 1

        return MGraphResults, self.patience

    def after_optim_step_MGraph(self):
        if self.MGraphScheduler.total_warmup_steps > self.MGraphScheduler._step:
            self.step_graph_schedulers()

    def step_graph_schedulers(self, metrics = None):
        try:
            self.MGraphScheduler.step(metrics = metrics)
        except:
            self.MGraphScheduler.step()

    def loss_cl(self, x1, x2):

        T = 0.1
        
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        # x1: (B, 256), x2: (B, 256)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs) # (B, B)
        sim_matrix = torch.exp(sim_matrix / T)
        
        pos_sim = sim_matrix[range(batch_size), range(batch_size)] # (B)

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        loss = torch.relu(loss)
        
        return loss
        
    def loss_tri(self, graph_rep, graph_rep1, graph_rep2):
        loss = self.triplet_loss(graph_rep, graph_rep1, graph_rep2)
        return loss
        
    def evaluate_metrics(self, predictions, targets):
        metrics = {}
        
        metrics[f'mean_pred'] = torch.mean(predictions).item()
        metrics[f'std_pred'] = torch.std(predictions).item()
        metrics[f'mean_targets'] = torch.mean(targets).item()
        metrics[f'std_targets'] = torch.std(targets).item()

        return metrics   
        
    def get_codebook(self, graph, masked_atom_indices, masked_node_labels, masked_edge_indices, masked_edge_labels):
        
        graph2 = copy.deepcopy(graph)
        node_logits, graph_logits = self.tokenizer(graph2)
        atom_ids = node_logits.argmax(dim = -1)
        
        return atom_ids  

def move_to_device(element, device):
    '''
    takes arbitrarily nested list and moves everything in it to device if it is a dgl graph or a torch tensor
    :param element: arbitrarily nested list
    :param device:
    :return:
    '''
    if isinstance(element, list):
        return tuple([move_to_device(x, device) for x in element])
    else:
        return element.to(device) if isinstance(element,(torch.Tensor, dgl.DGLGraph)) else element

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)
