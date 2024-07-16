import os
import dgl
import copy
import torch
import pickle
import random
import numpy as np
import pandas as pd
from typing import List, Tuple

from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

class MoleculeGraphDataset(Dataset):
    
    def __init__(self, processed_file, device='cuda:0'):
        self.processed_file = processed_file
        self.device = device

        print('Load data ...')
        if not os.path.exists(f"{self.processed_file}"):
            print(f"Check input file: {self.processed_file}")
        data_dict = torch.load(f"{self.processed_file}")
        
        self.features_tensor = data_dict['atom_features']
        self.e_features_tensor = data_dict['edge_features']
        self.edge_indices = data_dict['edge_indices']
        
        self.meta_dict = {k: data_dict[k] for k in ('mol_ids', 'edge_slices', 'atom_slices', 'n_atoms')}

    def __len__(self):
        return len(self.meta_dict['mol_ids'])
    
    def __getitem__(self, idx):
        data = list()
        e_start = self.meta_dict['edge_slices'][idx].item()
        e_end = self.meta_dict['edge_slices'][idx + 1].item()
        start = self.meta_dict['atom_slices'][idx].item()
        n_atoms = self.meta_dict['n_atoms'][idx].item()
        
        return self.data_by_type(e_start, e_end, start, n_atoms)

    def data_by_type(self, e_start, e_end, start, n_atoms):
        g = self.get_graph(e_start, e_end, n_atoms, start)
        return g

    def get_graph(self, e_start, e_end, n_atoms, start):
        edge_indices = self.edge_indices[:, e_start: e_end]
        g = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms, device=self.device)
        g.ndata['feat'] = self.features_tensor[start: start + n_atoms].to(self.device)
        g.edata['feat'] = self.e_features_tensor[e_start: e_end].to(self.device)
        return g

def graph_collate(batch):
    return dgl.batch(batch)

class DataLoaderMaskingPred(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, num_atom_type = 118, num_edge_type = 4, 
            mask_rate = 0.0, mask_edge = 0.0, **kwargs):
        
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        
        super(DataLoaderMaskingPred, self).__init__(
            dataset, batch_size, shuffle, collate_fn = self.collate_fn,
            **kwargs)
        
    def collate_fn(self, batch):

        batch_masked_atom_indices, batch_masked_edge_indices = list(), list()
        batch_mask_node_labels, batch_mask_edge_labels = list(), list()
        accum_node, accum_edge = 0, 0
        
        for graph in batch:
            ### masked atom random sampling
            num_atoms = graph.ndata['feat'].size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

            for atom_idx in masked_atom_indices:
                batch_mask_node_labels.append(copy.deepcopy(graph.ndata['feat'][atom_idx]).view(1, -1))

            # modify the orignal node feature of the masked node
            for atom_idx in masked_atom_indices:
                graph.ndata['feat'][atom_idx] = torch.tensor([self.num_atom_type, 0, 0, 0, 0, 0, 0, 0, 0]) # original atom type to masked token

            if self.mask_edge:
                # create mask edge labels by copying edge features of edges that are bonded to mask atoms
                connected_edge_indices = list()

                for bond_idx, (u, v) in enumerate(zip(graph.edges()[0].cpu().numpy(), graph.edges()[1].cpu().numpy())):
                    for atom_idx in masked_atom_indices:
                        if atom_idx in set((u, v)) and bond_idx not in connected_edge_indices:
                            connected_edge_indices.append(bond_idx)   

                if len(connected_edge_indices) > 0 :
                    # create mask edge labels by copying bond features of the bonds connected to the mask atoms
                    for bond_idx in connected_edge_indices[::2]: # because the
                        # edge ordering is such that two directions of a single
                        # edge occur in pairs, so to get the unique undirected
                        # edge indices, we take every 2nd edge index from list 
                        batch_mask_edge_labels.append(
                            copy.deepcopy(graph.edata['feat'][bond_idx]).view(1, -1))

                    for bond_idx in connected_edge_indices:
                        graph.edata['feat'][bond_idx] = torch.tensor(
                            [self.num_edge_type, 0, 0, 0])

                    masked_atom_indices = torch.tensor(masked_atom_indices)
                    connected_edge_indices = torch.tensor(connected_edge_indices[::2])

                    masked_atom_indices += accum_node
                    connected_edge_indices += accum_edge

                    accum_node += graph.ndata['feat'].size()[0]
                    accum_edge += graph.edata['feat'].size()[0]
                    
                    batch_masked_atom_indices.extend(masked_atom_indices)
                    batch_masked_edge_indices.extend(connected_edge_indices)
                    
        return [dgl.batch(batch), torch.tensor(batch_masked_atom_indices).to(torch.int64), torch.cat(batch_mask_node_labels, dim = 0), torch.tensor(batch_masked_edge_indices).to(torch.int64), torch.cat(batch_mask_edge_labels, dim = 0)]

def contrastive_collate(batch: List[Tuple]):
    # optionally take targets
    graphs, graphs3d, *targets = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    batched_graph3d = dgl.batch(graphs3d)
  
    return [batched_graph, batched_graph3d]