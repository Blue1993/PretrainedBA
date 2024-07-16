import yaml
import pprint
import os
import logging
import sys

import pandas as pd
import csv

import pickle
import numpy as np
import random 
import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    stream=sys.stdout,
)

def load_cfg(yaml_filepath):
    
    ### Load a yaml configuration file.
    """
    # Parameters
        - yaml file path: str
    # Returns
        - cfg: dict
    """
    
    with open(yaml_filepath, "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    
    return cfg
    
    #return Config(cfg)

def make_paths_absolute(dir_, cfg):
    
    ### Make all values for keys ending with '_path' absolute to dir_.
    
    """
    # Parameters
        - dir_: str
        - cfg: dict
    # Returns
        - cfg: dict
    """
    
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]) and not os.path.isdir(cfg[key]):
                logging.error("%s does not exist.", cfg[key])

        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    
    return cfg

class MaskAtom: 
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge = True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        """
        self.num_atom_type = num_atom_type # 120 (masked bond index is 119): 118 atom type + 1 ? + 1 masked atom
        self.num_edge_type = num_edge_type # 6 (masked bond index is 5): 4 bone type + 1 self loop + 1 masked bond
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
    def __call__(self, data, masked_atom_indices = None):
        """
        param data: pytroch geometric data object.
        Assume that the edge ordering is default pytorch geometric odering, where the two
        directions of a single edge occur in pairs.
        e.g., data.edge_index = tensor([[0, 1, 1, 2, 2, 3],[1, 0, 2, 1, 3, 2]])
        """
        
        ### masked atom radom sampling
        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. 
            # But will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)
            
        # create mask node label by copying atom feature of mask atom    
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)    
        data.masked_atom_indices = torch.tensor(masked_atom_indices)    
            
        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor([self.num_atom_type, 0]) # original atom type to masked token
        
        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)
