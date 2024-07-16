import numpy as np
import torch
import pickle
import os
import dgl
from torch.utils.data import Dataset



class TrainDataset(Dataset):
    def __init__(self, interaction_IDs, labels, device,
                    protein_features_path, pocket_indices_path, compound_features_path, interaction_sites_path):
        
        self.interaction_IDs = interaction_IDs
        self.labels = labels
        self.device = device
        
        self.protein_features_path = protein_features_path
        self.compound_features_path = compound_features_path
        self.interaction_sites_path = interaction_sites_path
        self.pocket_indices_path = pocket_indices_path

        print("Preparing protein data ...")
        with open(f"{self.protein_features_path}", "rb") as f:
            self.protein_data_dict = pickle.load(f)
        
        print(f"\tLoad No. of Protein: {len(self.protein_data_dict)}")

        print("Preparing pocket data ...")
        with open(f"{pocket_indices_path}", "rb") as f:
            self.pocket_indices_dict = pickle.load(f)
        print(f"\tLoad No. of Pockets: {len(self.pocket_indices_dict)}")    

        print("Preparing compound data ...")
        compound_data_dict = torch.load(f"{self.compound_features_path}")
        print(f"\tLoad No. of Compound: {len(compound_data_dict['mol_ids'])}")
        
        print("Preparing interaction data ...")
        if not os.path.exists(f"{self.interaction_sites_path[:-4]}.pt"):
            with open(f"{self.interaction_sites_path}", "rb") as f:
                interaction_site_data_dict = pickle.load(f)

            torch.save(interaction_site_data_dict, f"{self.interaction_sites_path[:-4]}.pt")
        self.interaction_site_data_dict = torch.load(f"{self.interaction_sites_path[:-4]}.pt")
        print(f"\tLoad No. of Interactions: {len(self.interaction_site_data_dict)}")
        
        self.compound_ids = compound_data_dict['mol_ids']
        self.compound_ids_dict = {molid:idx for idx, molid in enumerate(self.compound_ids)}
        self.compound_features_tensor = compound_data_dict['atom_features']
        self.compound_e_features_tensor = compound_data_dict['edge_features']
        self.edge_indices = compound_data_dict['edge_indices']
        
        self.compound_meta_dict = {k: compound_data_dict[k] for k in ('mol_ids', 'edge_slices', 'atom_slices', 'n_atoms')} 
        self.avg_degree = compound_data_dict['avg_degree']

    def __len__(self):
        return len(self.interaction_IDs)
    
    def __getitem__(self, idx):
        
        pdbid, pid, cid = self.interaction_IDs[idx].split("_")[0], self.interaction_IDs[idx].split("_")[1], self.interaction_IDs[idx].split("_")[2]
        label = self.labels[idx]
        
        #print(self.interaction_IDs[idx])
        
        ####################
        # Get Protein data
        ####################
        residue_feat = self.protein_data_dict[pid]
        seqlength = residue_feat.shape[0]
        pocket_index = self.pocket_indices_dict[pid]

        ####################
        # Get Compound data
        ####################
        comp_idx = self.compound_ids_dict[cid]
        e_start = self.compound_meta_dict['edge_slices'][comp_idx].item()
        e_end = self.compound_meta_dict['edge_slices'][comp_idx + 1].item()
        start = self.compound_meta_dict['atom_slices'][comp_idx].item()
        n_atoms = self.compound_meta_dict['n_atoms'][comp_idx].item()
        
        compound_graph = self.data_by_type(e_start, e_end, start, n_atoms)
        num_node = compound_graph.number_of_nodes()
        
        #######################
        # Get Interaction data
        #######################
        interaction_site = self.interaction_site_data_dict[f"{pdbid}_{cid}"]

        return {"residue_feat":residue_feat, "seqlength": seqlength, "pocket_index":pocket_index,
                    "compound_graph":compound_graph, "num_node":num_node,
                        "interaction_site":interaction_site, "label":label}

    def data_by_type(self, e_start, e_end, start, n_atoms):
        g = self.get_graph(e_start, e_end, n_atoms, start)
        return g
    
    def get_graph(self, e_start, e_end, n_atoms, start):
        edge_indices = self.edge_indices[:, e_start: e_end]
        g = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms, device=self.device)
        g.ndata['feat'] = self.compound_features_tensor[start: start + n_atoms].to(self.device)
        g.edata['feat'] = self.compound_e_features_tensor[e_start: e_end].to(self.device)
        #self.dgl_graphs[idx] = g.to('cpu')
        return g

def pad_data(samples):
    
    # protein features
    residue_feats = [sample['residue_feat'] for sample in samples]
    pocket_indices = [sample['pocket_index'] for sample in samples]
    seqlengths = [sample['seqlength'] for sample in samples]
    pocket_seqlengths = [len(sample) for sample in pocket_indices]
    
    # compound features
    compound_graphs = [sample['compound_graph'] for sample in samples]
    num_nodes = [sample['num_node'] for sample in samples]
    
    # interaction site labels
    interaction_sites = [sample['interaction_site'] for sample in samples]
    
    # ba labels
    labels = [sample['label'] for sample in samples]
    
    # prepare input representations
    batch_size, protein_max_seq_length, pocket_max_seq_length, compound_max_atom_count = len(residue_feats), np.max(seqlengths), np.max(pocket_seqlengths), np.max(num_nodes)
    
    for i in pocket_seqlengths:
        if i == 0:
            pocket_max_seq_length = protein_max_seq_length
    
    protein_mask = np.zeros((batch_size, protein_max_seq_length))
    pocket_mask, compound_mask, interaction_labels = np.zeros((batch_size, pocket_max_seq_length)), np.zeros((batch_size, compound_max_atom_count)), np.zeros((batch_size, compound_max_atom_count, protein_max_seq_length))
    pocket_interaction_labels = np.zeros((batch_size, compound_max_atom_count, pocket_max_seq_length))

    input_residue_feats = np.zeros((batch_size, pocket_max_seq_length, 1024))
    input_pocket_indices = np.zeros((batch_size, pocket_max_seq_length)) - 1
    
    # extract pocket residue info
    for idx, arr in enumerate(residue_feats):
        protein_mask[idx, :arr.shape[0]] = 1
        
        if pocket_seqlengths[idx] == 0:
            extract_arr = arr[:, :]
            input_residue_feats[idx, :extract_arr.shape[0], :] = extract_arr
            
            pocket_mask[idx, :extract_arr.shape[0]] = 1
            
        else:
            extract_arr = arr[pocket_indices[idx],:]
            input_residue_feats[idx, :extract_arr.shape[0], :] = extract_arr
            
            pocket_mask[idx, :extract_arr.shape[0]] = 1
            input_pocket_indices[idx, :pocket_seqlengths[idx]] = pocket_indices[idx]
    
    # make compound mask
    for idx, n_node in enumerate(num_nodes):
        compound_mask[idx, :n_node] = 1

    # extract interaction sites info with pocket residue
    for idx, arr in enumerate(interaction_sites):
        isarr = arr[1]
        interaction_labels[idx, :isarr.shape[0], :isarr.shape[1]] = isarr
        
        if pocket_seqlengths[idx] == 0:
            extract_isarr = isarr[:,:]
            pocket_interaction_labels[idx, :extract_isarr.shape[0], :extract_isarr.shape[1]] = extract_isarr
            
        else:
            extract_isarr = isarr[:,pocket_indices[idx]]
            pocket_interaction_labels[idx, :extract_isarr.shape[0], :extract_isarr.shape[1]] = extract_isarr
        
    # pairwise mask 
    pairwise_mask = np.matmul(np.expand_dims(compound_mask, axis = 2), np.expand_dims(protein_mask, axis = 1))
    
    # convert type for input data
    input_residue_feats = torch.tensor(input_residue_feats, dtype = torch.float32).cuda()
    input_pocket_indices = torch.tensor(input_pocket_indices, dtype = torch.long).cuda()
    seqlengths = torch.tensor(seqlengths, dtype = torch.long).cuda()
    
    labels = torch.tensor(labels, dtype = torch.float32).cuda()
    
    pocket_interaction_labels = torch.tensor(pocket_interaction_labels, dtype = torch.float32).cuda()
    interaction_labels = torch.tensor(interaction_labels, dtype = torch.float32).cuda()
    
    pairwise_mask = torch.tensor(pairwise_mask, dtype = torch.long).cuda()
    pocket_mask = torch.tensor(pocket_mask, dtype = torch.long).cuda()
    compound_mask = torch.tensor(compound_mask, dtype = torch.long).cuda()

    return (input_residue_feats, input_pocket_indices, seqlengths), dgl.batch(compound_graphs).to("cuda:0"), labels, pocket_interaction_labels, interaction_labels, pairwise_mask, pocket_mask, compound_mask 

'''
def pad_data(samples):
    
    # protein features
    residue_feats = [sample['residue_feat'] for sample in samples]
    pocket_indices = [sample['pocket_index'] for sample in samples]
    seqlengths = [sample['seqlength'] for sample in samples]
    pocket_seqlengths = [len(sample) for sample in pocket_indices]
    
    # compound features
    compound_graphs = [sample['compound_graph'] for sample in samples]
    num_nodes = [sample['num_node'] for sample in samples]
    
    # interaction site labels
    interaction_sites = [sample['interaction_site'] for sample in samples]
    
    # ba labels
    labels = [sample['label'] for sample in samples]
    
    # prepare input representations
    batch_size, protein_max_seq_length, pocket_max_seq_length, compound_max_atom_count = len(residue_feats), np.max(seqlengths), np.max(pocket_seqlengths), np.max(num_nodes)
    protein_mask = np.zeros((batch_size, protein_max_seq_length))
    pocket_mask, compound_mask, interaction_labels = np.zeros((batch_size, pocket_max_seq_length)), np.zeros((batch_size, compound_max_atom_count)), np.zeros((batch_size, compound_max_atom_count, protein_max_seq_length))
    pocket_interaction_labels = np.zeros((batch_size, compound_max_atom_count, pocket_max_seq_length))

    input_residue_feats = np.zeros((batch_size, pocket_max_seq_length, 1024))
    input_pocket_indices = np.zeros((batch_size, pocket_max_seq_length)) - 1
    
    # extract pocket residue info
    for idx, arr in enumerate(residue_feats):
        protein_mask[idx, :arr.shape[0]] = 1
        
        extract_arr = arr[pocket_indices[idx],:]
        input_residue_feats[idx, :extract_arr.shape[0], :] = extract_arr
        
        pocket_mask[idx, :extract_arr.shape[0]] = 1
        input_pocket_indices[idx, :pocket_seqlengths[idx]] = pocket_indices[idx]
    
    # make compound mask
    for idx, n_node in enumerate(num_nodes):
        compound_mask[idx, :n_node] = 1

    # extract interaction sites info with pocket residue
    for idx, arr in enumerate(interaction_sites):
        isarr = arr[1]
        interaction_labels[idx, :isarr.shape[0], :isarr.shape[1]] = isarr
        
        extract_isarr = isarr[:,pocket_indices[idx]]
        pocket_interaction_labels[idx, :extract_isarr.shape[0], :extract_isarr.shape[1]] = extract_isarr
    
    # pairwise mask 
    pairwise_mask = np.matmul(np.expand_dims(compound_mask, axis = 2), np.expand_dims(protein_mask, axis = 1))
    
    # convert type for input data
    input_residue_feats = torch.tensor(input_residue_feats, dtype = torch.float32).cuda()
    input_pocket_indices = torch.tensor(input_pocket_indices, dtype = torch.long).cuda()
    seqlengths = torch.tensor(seqlengths, dtype = torch.long).cuda()
    
    labels = torch.tensor(labels, dtype = torch.float32).cuda()
    
    pocket_interaction_labels = torch.tensor(pocket_interaction_labels, dtype = torch.float32).cuda()
    interaction_labels = torch.tensor(interaction_labels, dtype = torch.float32).cuda()
    
    pairwise_mask = torch.tensor(pairwise_mask, dtype = torch.long).cuda()
    pocket_mask = torch.tensor(pocket_mask, dtype = torch.long).cuda()
    compound_mask = torch.tensor(compound_mask, dtype = torch.long).cuda()

    return (input_residue_feats, input_pocket_indices, seqlengths), dgl.batch(compound_graphs).to("cuda:0"), labels, pocket_interaction_labels, interaction_labels, pairwise_mask, pocket_mask, compound_mask 
'''
class TestDataset(Dataset):
    def __init__(self, interaction_IDs, labels, device,
                protein_features_path, pocket_indices_path, compound_features_path):

        self.interaction_IDs = interaction_IDs
        self.labels = labels
        self.device = device    

        self.protein_features_path = protein_features_path
        self.compound_features_path = compound_features_path
        self.pocket_indices_path = pocket_indices_path

        print(f"\tLoad No. of Interaction: {len(self.interaction_IDs)}")
        
        print("Preparing protein data ...")
        with open(f"{self.protein_features_path}", "rb") as f:
            self.protein_data_dict = pickle.load(f)
        print(f"\tLoad No. of Protein: {len(self.protein_data_dict)}")
        
        print("Preparing pocket data ...")
        with open(f"{pocket_indices_path}", "rb") as f:
            self.pocket_indices_dict = pickle.load(f)
        print(f"\tLoad No. of Proteins (Pockets): {len(self.pocket_indices_dict)}")    
        
        print("Preparing compound data ...")
        compound_data_dict = torch.load(f"{self.compound_features_path}")
        print(f"\tLoad No. of Compound: {len(compound_data_dict['mol_ids'])}")
        
        self.compound_ids = compound_data_dict['mol_ids']
        self.compound_id_dict = {molid:idx for idx, molid in enumerate(self.compound_ids)}
        self.compound_features_tensor = compound_data_dict['atom_features']
        self.compound_e_features_tensor = compound_data_dict['edge_features']
        self.edge_indices = compound_data_dict['edge_indices']
        
        self.compound_meta_dict = {k: compound_data_dict[k] for k in ('mol_ids', 'edge_slices', 'atom_slices', 'n_atoms')} 
        self.avg_degree = compound_data_dict['avg_degree']
        #self.dgl_graphs = {}
        
    def __len__(self):
        return len(self.interaction_IDs)
        
    def __getitem__(self, idx):

        pid, cid = self.interaction_IDs[idx].split("_")[0], self.interaction_IDs[idx].split("_")[1]
        label = self.Labels[idx]
        
        ####################
        # Get Protein data
        ####################
        residue_feat = self.protein_data_dict[pid]
        seqlength = residue_feat.shape[0]
        pocket_index = self.pocket_indices_dict[pid]

        ####################
        # Get Compound data
        ####################
        comp_idx = self.compound_id_dict[cid]
        e_start = self.compound_meta_dict['edge_slices'][comp_idx].item()
        e_end = self.compound_meta_dict['edge_slices'][comp_idx + 1].item()
        start = self.compound_meta_dict['atom_slices'][comp_idx].item()
        n_atoms = self.compound_meta_dict['n_atoms'][comp_idx].item()
        
        compound_graph = self.data_by_type(e_start, e_end, start, n_atoms)
        num_node = compound_graph.number_of_nodes()

        return {"pfeat":pfeat, "seqlength": seqlength, "pocket":pocket,
                    "compound_graph":compound_graph, "num_node":num_node, "label":label}

    def data_by_type(self, e_start, e_end, start, n_atoms):
        g = self.get_graph(e_start, e_end, n_atoms, start)
        return g
    
    def get_graph(self, e_start, e_end, n_atoms, start):
        edge_indices = self.edge_indices[:, e_start: e_end]
        g = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms, device=self.device)
        g.ndata['feat'] = self.compound_features_tensor[start: start + n_atoms].to(self.device)
        g.edata['feat'] = self.compound_e_features_tensor[e_start: e_end].to(self.device)
        return g

def pad_data_for_test(samples):
    
    # protein features
    residue_feats = [sample['residue_feat'] for sample in samples]
    pocket_indices = [sample['pocket_index'] for sample in samples]
    seqlengths = [sample['seqlength'] for sample in samples]
    pocket_seqlengths = [len(sample) for sample in pocket_indices]
    
    # compound features
    compound_graphs = [sample['compound_graph'] for sample in samples]
    num_nodes = [sample['num_node'] for sample in samples]
    
    # ba labels
    labels = [sample['label'] for sample in samples]
    
    # prepare input representations
    batch_size, protein_max_seq_length, pocket_max_seq_length, compound_max_atom_count = len(residue_feats), np.max(seqlengths), np.max(pocket_seqlengths), np.max(num_nodes)
    protein_mask = np.zeros((batch_size, protein_max_seq_length))
    pocket_mask, compound_mask = np.zeros((batch_size, pocket_max_seq_length)), np.zeros((batch_size, compound_max_atom_count))

    input_residue_feats = np.zeros((batch_size, pocket_max_seq_length, 1024))
    input_pocket_indices = np.zeros((batch_size, pocket_max_seq_length)) - 1
    
    # extract pocket residue info
    for idx, arr in enumerate(residue_feats):
        protein_mask[idx, :arr.shape[0]] = 1
        
        extract_arr = arr[pocket_indices[idx],:]
        input_residue_feats[idx, :extract_arr.shape[0], :] = extract_arr
        
        pocket_mask[idx, :extract_arr.shape[0]] = 1
        input_pocket_indices[idx, :pocket_seqlengths[idx]] = pocket_indices[idx]
    
    # make compound mask
    for idx, n_node in enumerate(num_nodes):
        compound_mask[idx, :n_node] = 1

    # convert input data type
    input_residue_feats = torch.tensor(input_residue_feats, dtype = torch.float32).cuda()
    input_pocket_indices = torch.tensor(input_pocket_indices, dtype = torch.long).cuda()
    seqlengths = torch.tensor(seqlengths, dtype = torch.long).cuda()
    
    labels = torch.tensor(labels, dtype = torch.float32).cuda()

    pocket_mask = torch.tensor(pocket_mask, dtype = torch.long).cuda()
    compound_mask = torch.tensor(compound_mask, dtype = torch.long).cuda()

    return (input_residue_feats, input_pocket_indices, seqlengths), dgl.batch(compound_graphs).to("cuda:0"), labels, pocket_mask, compound_mask 
    