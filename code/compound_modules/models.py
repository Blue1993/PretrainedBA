import dgl
from math import sqrt
from typing import List
import dgl.function as fn
from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, softmax

from compound_modules.base_layers import MLP

num_atom_type = 119 # do not use

# use atom number (1~118), masked token number: 119; so 120 atom type; 0 is not used
num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
# 0: single, 1: double, 2: triple, 3: aromatic, 4: self-loop, 5: masked
num_bond_direction = 6 

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
    
    'possible_bond_dirs' : [ 
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.BEGINDASH, # new add
        Chem.rdchem.BondDir.BEGINWEDGE, # new add
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        Chem.rdchem.BondDir.EITHERDOUBLE, # new add
    ]
}

class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized
        - embedding_dim (int): the dimensionality of the tensors in the quantized space. Inputs to the modules must be in this format as well.
        - num_embeddings (int): the number of vectors in the quantized space
        - commitment_cost (float): scaler which controls the weighting of the the loss terms
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim # (256)
        self.num_embeddings = num_embeddings # (512)
        self.commitment_cost = commitment_cost # (2.0)
        
        #initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim) # (512, 256)
        
    def forward(self, x, e): # (x: atom_features (B, N, 9), e: node_embeddings (B, N, 256))
        encoding_indices = self.get_code_indices(x, e) # x: B * H, encoding_indices: B

        quantized = self.quantize(encoding_indices)

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, e.detach())

        # commitment loss
        e_latent_loss = F.mse_loss(e, quantized.detach())

        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = e + (quantized - e).detach().contiguous()
        return quantized, loss
    
    def get_code_indices(self, x, e):
        # x: N * 2  e: N * E
        atom_type = x[:, 0]
        index_c = (atom_type == 5)
        index_n = (atom_type == 6)
        index_o = (atom_type == 7)
        index_others = ~(index_c + index_n + index_o)
        # compute L2 distance
        encoding_indices = torch.ones(x.size(0)).long().to(x.device)
        # C:
        distances = (
            torch.sum(e[index_c] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[0: 377] ** 2, dim=1) -
            2. * torch.matmul(e[index_c], self.embeddings.weight[0: 377].t())
        )
        encoding_indices[index_c] = torch.argmin(distances, dim=1)
        # N:
        distances = (
            torch.sum(e[index_n] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[378: 433] ** 2, dim=1) -
            2. * torch.matmul(e[index_n], self.embeddings.weight[378: 433].t())
        ) 
        encoding_indices[index_n] = torch.argmin(distances, dim=1) + 378
        # O:
        distances = (
            torch.sum(e[index_o] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[434: 488] ** 2, dim=1) -
            2. * torch.matmul(e[index_o], self.embeddings.weight[434: 488].t())
        )   
        encoding_indices[index_o] = torch.argmin(distances, dim=1) + 434

        # Others:
        distances = (
            torch.sum(e[index_others] ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight[489: 511] ** 2, dim=1) -
            2. * torch.matmul(e[index_others], self.embeddings.weight[489: 511].t())
        ) 
        encoding_indices[index_others] = torch.argmin(distances, dim=1) + 489

        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices) 

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file)) 


class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super().__init__()
        self._dec_type = gnn_type 
        if gnn_type == "gin":
            self.conv = GINConv(hidden_dim, out_dim, aggr = "add")
        #elif gnn_type == "gcn":
        #    self.conv = GCNConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "linear":
            self.dec = torch.nn.Linear(hidden_dim, out_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")
        #self.dec_token = torch.nn.Parameter(torch.zeros([1, hidden_dim])) # (1, 300)
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False) # (300, 300)
        self.activation = torch.nn.PReLU() 
        #self.activation = torch.nn.SiLU() 
        #self.activation = torch.nn.ReLU()
        #self.temp = 0.2

    def forward(self, x, edge_index, edge_attr):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            x = self.activation(x)
            x = self.enc_to_dec(x)
            # x[mask_node_indices] = 0
            # x[mask_node_indices] = self.dec_token
            out = self.conv(x, edge_index, edge_attr)
            # out = F.softmax(out, dim=-1) / self.temp
        return out

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation
    
    Args
        - emb_dim (int): dimensionality of embeddings for nodes and edges
        - embed_input (bool): whether to embed input or not
    
    See https://arxiv.org/abs/1810.00826
    """
    
    def __init__(self, emb_dim, out_dim, aggr = "add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        # multi-layer perceptron
        #self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, out_dim))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.SiLU(), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim) # (6, 300)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim) # (6, 300)
        
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
    
    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        #self_loop_attr = torch.zeros(x.size(0), 4)

        #self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr[:,0] = 5 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings) # when call propagate -> call message() and upate () call Sequentially
    
    # Ref: https://greeksharifa.github.io/pytorch/2021/09/04/MP/
    def message(self, x_j, edge_attr): # The part that defines how to process the information of neighboring node x_j and deliver it to target node x_i
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class Net3D(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, target_dim, readout_aggregators: List[str], batch_norm=False,
                 node_wise_output_layers=2, readout_batchnorm=True, batch_norm_momentum=0.1, reduce_func='sum',
                 dropout=0.0, propagation_depth: int = 4, readout_layers: int = 2, readout_hidden_dim=None,
                 fourier_encodings=0, activation: str = 'SiLU', update_net_layers=2, message_net_layers=2, use_node_features=False, **kwargs):
        super(Net3D, self).__init__()
        self.fourier_encodings = fourier_encodings # 4
        edge_in_dim = 1 if fourier_encodings == 0 else 2 * fourier_encodings + 1 # 9
        self.edge_input = MLP(in_dim=edge_in_dim, hidden_size=hidden_dim, out_dim=hidden_dim, mid_batch_norm=batch_norm,
                              last_batch_norm=batch_norm, batch_norm_momentum=batch_norm_momentum, layers=1,
                              mid_activation=activation, dropout=dropout, last_activation=activation,
                              )
        
        self.use_node_features = use_node_features # None
        if self.use_node_features:
            self.atom_encoder = AtomEncoder(hidden_dim)
        else:
            self.node_embedding = nn.Parameter(torch.empty((hidden_dim,)))
            nn.init.normal_(self.node_embedding)

        self.mp_layers = nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(
                Net3DLayer(edge_dim=hidden_dim, hidden_dim=hidden_dim, batch_norm=batch_norm,
                           batch_norm_momentum=batch_norm_momentum, dropout=dropout, mid_activation=activation,
                           reduce_func=reduce_func, message_net_layers=message_net_layers,
                           update_net_layers=update_net_layers))

        self.node_wise_output_layers = node_wise_output_layers
        if self.node_wise_output_layers > 0:
            self.node_wise_output_network = MLP(in_dim=hidden_dim, hidden_size=hidden_dim, out_dim=hidden_dim,
                                                mid_batch_norm=batch_norm, last_batch_norm=batch_norm,
                                                batch_norm_momentum=batch_norm_momentum, layers=node_wise_output_layers,
                                                mid_activation=activation, dropout=dropout, last_activation='None',
                                                )

        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, batch_norm_momentum=batch_norm_momentum, out_dim=target_dim,
                          layers=readout_layers)

    def forward(self, graph: dgl.DGLGraph):
        if self.use_node_features:
            graph.ndata['feat'] = self.atom_encoder(graph.ndata['feat'])
        else:
            graph.ndata['feat'] = self.node_embedding[None, :].expand(graph.number_of_nodes(), -1)

        if self.fourier_encodings > 0:
            graph.edata['d'] = fourier_encode_dist(graph.edata['d'], num_encodings=self.fourier_encodings)
        graph.apply_edges(self.input_edge_func)

        for mp_layer in self.mp_layers:
            mp_layer(graph)

        if self.node_wise_output_layers > 0:
            graph.apply_nodes(self.output_node_func)

        readouts_to_cat = [dgl.readout_nodes(graph, 'feat', op=aggr) for aggr in self.readout_aggregators]

        readout = torch.cat(readouts_to_cat, dim=-1)

        return self.output(readout)

    def output_node_func(self, nodes):
        return {'feat': self.node_wise_output_network(nodes.data['feat'])}

    def input_edge_func(self, edges):
        return {'d': F.silu(self.edge_input(edges.data['d']))}


class Net3DLayer(nn.Module):
    def __init__(self, edge_dim, reduce_func, hidden_dim, batch_norm, batch_norm_momentum, dropout,
                 mid_activation, message_net_layers, update_net_layers):
        super(Net3DLayer, self).__init__()
        self.message_network = MLP(in_dim=hidden_dim * 2 + edge_dim, hidden_size=hidden_dim, out_dim=hidden_dim,
                                   mid_batch_norm=batch_norm, last_batch_norm=batch_norm,
                                   batch_norm_momentum=batch_norm_momentum, layers=message_net_layers,
                                   mid_activation=mid_activation, dropout=dropout, last_activation=mid_activation,
                                   )
        if reduce_func == 'sum':
            self.reduce_func = fn.sum
        elif reduce_func == 'mean':
            self.reduce_func = fn.mean
        else:
            raise ValueError('reduce function not supported: ', reduce_func)

        self.update_network = MLP(in_dim=hidden_dim, hidden_size=hidden_dim, out_dim=hidden_dim,
                                  mid_batch_norm=batch_norm, last_batch_norm=batch_norm,
                                  batch_norm_momentum=batch_norm_momentum, layers=update_net_layers,
                                  mid_activation=mid_activation, dropout=dropout, last_activation='None',
                                  )

        self.soft_edge_network = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        graph.update_all(message_func=self.message_function, reduce_func=self.reduce_func(msg='m', out='m_sum'),
                         apply_node_func=self.update_function)

    def message_function(self, edges):
        message_input = torch.cat(
            [edges.src['feat'], edges.dst['feat'], edges.data['d']], dim=-1)
        message = self.message_network(message_input)
        edges.data['d'] += message
        edge_weight = torch.sigmoid(self.soft_edge_network(message))
        return {'m': message * edge_weight}

    def update_function(self, nodes):
        h = nodes.data['feat']
        input = torch.cat([nodes.data['m_sum'] + nodes.data['feat']], dim=-1)
        h_new = self.update_network(input)
        output = h_new + h
        return {'feat': output}

def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x.squeeze()

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, padding=False):
        """
        :param emb_dim: the dimension that the returned embedding will have
        :param padding: if this is true then -1 will be mapped to padding
        """
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.padding = padding
        full_atom_feature_dims = get_atom_feature_dims()
        
        ##############################
        """ atom embedding size """
        # - atomic_num: 119
        # - chirality: 5
        # - degree: 12
        # - formal_charge: 12
        # - numH: 10
        # - num_radical_e: 6
        # - hybridization: 6
        # - is_in_aromatic: 2
        # - is_in_ring: 2
        ##############################
        for i, dim in enumerate(full_atom_feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def reset_parameters(self):
        for i, embedder in enumerate(self.atom_embedding_list):
            embedder.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            if self.padding:
                #x_embedding += self.atom_embedding_list[i](x[:, i] + 1)
                x_embedding = x_embedding + self.atom_embedding_list[i](x[:, i] + 1)
            else:
                #x_embedding += self.atom_embedding_list[i](x[:, i])
                x_embedding = x_embedding + self.atom_embedding_list[i](x[:, i])
        return x_embedding

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list'],
        allowable_features['possible_bond_dirs']
        ]))

def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))

'''
def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list']
        #allowable_features['possible_is_conjugated_list'],
        #allowable_features['possible_bond_dirs']
        ]))

def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list']
        #allowable_features['possible_degree_list'],
        #allowable_features['possible_formal_charge_list'],
        #allowable_features['possible_numH_list'],
        #allowable_features['possible_number_radical_e_list'],
        #allowable_features['possible_hybridization_list'],
        #allowable_features['possible_is_aromatic_list'],
        #allowable_features['possible_is_in_ring_list']
        ]))
'''
class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim, padding=False):
        """
        :param emb_dim: the dimension that the returned embedding will have
        :param padding: if this is true then -1 will be mapped to padding
        """
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        self.padding = padding
        full_bond_feature_dims = get_bond_feature_dims()
        
        ##############################
        """ bond embedding size """
        # - possible_bond_type_list: 5
        # - possible_bond_stereo_list: 6 
        # - possible_is_conjugated_list: 2 
        # - possible_bond_dirs: 6
        ##############################
        
        for i, dim in enumerate(full_bond_feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            if self.padding:
                #bond_embedding += self.bond_embedding_list[i](edge_attr[:, i] + 1)
                bond_embedding = bond_embedding + self.bond_embedding_list[i](edge_attr[:, i] + 1)
            else:
                #bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])
                bond_embedding = bond_embedding + self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


























