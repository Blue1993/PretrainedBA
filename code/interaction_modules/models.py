import torch
import torch.nn as nn
import numpy as np
import sys
from os import path
import dgl
import math

sys.path.append(path.dirname(path.dirname(path.abspath(__file__) ) ))
from compound_modules.pna import PNA
from protein_modules.models import Pseq2Sites, ProtsEncoder

SUPPORTED_ACTIVATION_MAP = {'PReLU', 'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'SiLU', 'None'}
EPS = 1e-5

def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()

BertLayerNorm = torch.nn.LayerNorm

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class ProteinEncoderModule(nn.Module):
    def __init__(self, config, device):
        super(ProteinEncoderModule, self).__init__()
        
        self.config = config
        self.device = device
        
        self.self_attention = BertSelfattLayer(self.config)
        
        self.linear = nn.Linear(1024, 256)
        self.activation = torch.nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, protein_features, pocket_attention_mask, pocket_mask):
        
        protein_features = self.linear(protein_features)
        protein_features = torch.multiply(protein_features, pocket_mask.unsqueeze(2))
        protein_features = self.activation(protein_features)
        protein_features = self.dropout(protein_features)
        
        protein_features = self.self_attention(protein_features, pocket_attention_mask, pocket_mask)
        protein_features = torch.multiply(protein_features, pocket_mask.unsqueeze(2))  
        
        return protein_features

class PretrainedBA(nn.Module):
    def __init__(self, config, device):
        super(PretrainedBA, self).__init__()
        
        self.config = config
        self.device = device
        self.dropout_dta = nn.Dropout(0.2)
        
        # Define compound encoder
        self.compound_encoder = PNA(**config["MGraphModel"]["Architecture"]).to(self.device)

        # Define protein encoder
        self.protein_encoder = ProteinEncoderModule(self.config["CrossAttention"], self.device)

        # Define Cross Encoder
        self.cross_encoder = CrossAttentionModule(self.config["CrossAttention"])
        
        # Define Interaction Sites Module
        self.intersites_predictor = InteractionSitesModule(config["InteractionSite"]["Architecture"], self.device)
        
        # Define ba predictor
        self.ba_predictor =  nn.Sequential(
                        nn.Linear(128, 256),
                        torch.nn.BatchNorm1d(256),
                        nn.GELU(),
                        self.dropout_dta,
                        nn.Linear(256, 64),
                        torch.nn.BatchNorm1d(64),
                        nn.GELU(),
                        self.dropout_dta,
                        nn.Linear(64, 1)  
                    ) 
               
    def pad_to_graph(self, node_rep, num_node_list):
        total_node = 0
        max_node = torch.max(num_node_list)
        node_representation = torch.zeros(num_node_list.size()[0], max_node, node_rep.size()[1])
        
        for idx, n_atoms in enumerate(num_node_list):
            node_representation[idx, :n_atoms, :] = node_rep[total_node:total_node+n_atoms, :]
            total_node += n_atoms
        
        return node_representation.cuda()
        
    def forward(self, protein_data, compound_graphs, pocket_mask, compound_mask):
        
        # Get masks
        compound_mask = compound_mask.to(dtype=next(self.parameters()).dtype)
        pocket_mask = pocket_mask.to(dtype=next(self.parameters()).dtype)
        pairwise_mask = torch.matmul(compound_mask.unsqueeze(2), pocket_mask.unsqueeze(1)) 
        
        pocket_extend_mask = pocket_mask.unsqueeze(1).unsqueeze(2)
        pocket_extend_mask = pocket_extend_mask.to(dtype=next(self.parameters()).dtype) 
        pocket_extend_mask = (1.0 - pocket_extend_mask) * -1e9
        
        compound_extended_attention_mask = compound_mask.unsqueeze(1).unsqueeze(2) 
        compound_extended_attention_mask = compound_extended_attention_mask.to(dtype=next(self.parameters()).dtype) 
        compound_extended_attention_mask = (1.0 - compound_extended_attention_mask) * -1e9
        
        # get protein features
        resi_rep = self.protein_encoder(protein_data, pocket_extend_mask, pocket_mask)

        # get compound features
        node_rep, graph_rep = self.compound_encoder(compound_graphs) 
        node_rep = self.pad_to_graph(node_rep, compound_graphs.batch_num_nodes())
        
        # get cross features
        resi_rep, node_rep = self.cross_encoder(resi_rep, pocket_extend_mask, node_rep, compound_extended_attention_mask, pocket_mask, compound_mask) # resi rep: (batch, Max P, 256), node rep: (batch, Max A, 256)

        interaction_latent_vectors, pairwise_map_prediction = self.intersites_predictor(resi_rep, pocket_mask, node_rep, compound_mask, pairwise_mask)

        # get ba predictions
        outputs = self.ba_predictor(interaction_latent_vectors)
        return outputs.squeeze(1), pairwise_map_prediction, pairwise_mask

    def from_pretrained_model(self, model, model_path):    
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()

class InteractionSitesModule(nn.Module):
    def __init__(self, config, device):
        super(InteractionSitesModule, self).__init__()

        self.config = config
        self.device = device
        
        self.pairwise_compound = nn.Linear(self.config["hidden_size"], 64)
        self.pairwise_protein = nn.Linear(self.config["hidden_size"], 64)

        self.latent_compound = nn.Linear(self.config["hidden_size"], 64)
        self.latent_protein = nn.Linear(self.config["hidden_size"], 64)  
        
        self.activation = torch.nn.GELU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, protein_features, pocket_mask, compound_features, compound_mask, pairwise_mask):
        
        # get pairwise features
        pairwise_compound_features = self.activation(self.pairwise_compound(compound_features))
        pairwise_compound_features = self.dropout(pairwise_compound_features)
        pairwise_compound_features = torch.multiply(pairwise_compound_features, compound_mask.unsqueeze(2))

        pairwise_protein_features = self.activation(self.pairwise_protein(protein_features))
        pairwise_protein_features = self.dropout(pairwise_protein_features)
        pairwise_protein_features = torch.multiply(pairwise_protein_features, pocket_mask.unsqueeze(2))
        
        # get latent features
        latent_compound_features = self.activation(self.latent_compound(compound_features))
        latent_compound_features = self.dropout(latent_compound_features)
        latent_compound_features = torch.multiply(latent_compound_features, compound_mask.unsqueeze(2)) 

        latent_protein_feature = self.activation(self.latent_protein(protein_features))
        latent_protein_feature = self.dropout(latent_protein_feature)
        latent_protein_feature = torch.multiply(latent_protein_feature, pocket_mask.unsqueeze(2))
        
        # get pairwise map predictions
        pairwise_map = torch.matmul(pairwise_compound_features, pairwise_protein_features.transpose(1,2))

        # get latent vectors
        latent_features = torch.cat((latent_compound_features.unsqueeze(2).repeat(1, 1, protein_features.size()[1], 1), latent_protein_feature.unsqueeze(1).repeat(1, compound_features.size()[1], 1, 1)), dim = 3) 

        latent_features = torch.multiply(latent_features, pairwise_map.unsqueeze(3))
        latent_features = torch.multiply(latent_features, pairwise_mask.unsqueeze(3)) 
        
        latent_vectors = torch.sum(latent_features.view(latent_features.size()[0], latent_features.size()[1] * latent_features.size()[2], -1), dim = 1)

        pairwise_map = torch.sigmoid(pairwise_map)
        pairwise_map = pairwise_map*torch.matmul(compound_mask.unsqueeze(2), pocket_mask.unsqueeze(1)) 

        return latent_vectors, pairwise_map

class CrossAttentionModule(nn.Module):
    def __init__(self, config):
        super(CrossAttentionModule, self).__init__()
        self.config = config 

        self.cross_att_layers = nn.ModuleList(
            [CrossAttLayer(self.config) for _ in range(self.config["Architecture"]["num_layer"])]
        )
        
    def forward(self, protein_features, pocket_attention_mask, compound_features, compound_attention_mask, pocket_mask, compound_mask):

        for layer_module in self.cross_att_layers:
            protein_features, pocket_attention_mask, compound_features, compound_attention_mask = layer_module(protein_features, pocket_attention_mask, compound_features, compound_attention_mask, pocket_mask, compound_mask)

        return protein_features, compound_features

class CrossAttLayer(nn.Module):
    def __init__(self, config):
        super(CrossAttLayer, self).__init__()
        
        self.config = config
        
        # Cross-attention layer
        self.cross_attention = BertCrossattLayer(self.config)
        
        # Self-attention layer
        self.self_attention = BertSelfattLayer(self.config)
        
        # Intermediate and output layers (FFNs)
        self.inter = BertIntermediate(self.config)
        self.output = BertOutput(self.config)
        
    def cross_att(self, protein_inputs, pocket_attention_mask, compound_inputs, compound_attention_mask, pocket_mask, compound_mask):

        # Prots Cross Attention
        protein_att_outputs, _ = self.cross_attention(protein_inputs, compound_inputs, compound_attention_mask, pocket_mask)

        # Comps Cross Attention
        compound_att_outputs, _ = self.cross_attention(compound_inputs, protein_inputs, pocket_attention_mask, compound_mask)

        return protein_att_outputs, compound_att_outputs

    def self_att(self, protein_inputs, pocket_attention_mask, compound_inputs, compound_attention_mask, pocket_mask, compound_mask):
        protein_att_outputs = self.self_attention(protein_inputs, pocket_attention_mask, pocket_mask)
        compound_att_outputs = self.self_attention(compound_inputs, compound_attention_mask, compound_mask)
        return protein_att_outputs, compound_att_outputs

    def output_fc(self, protein_inputs, compound_inputs, pocket_mask, compound_mask):
        # FC layers
        protein_inter_outputs = self.inter(protein_inputs)
        compound_inter_outputs = self.inter(compound_inputs)

        protein_inter_outputs = torch.multiply(protein_inter_outputs, pocket_mask.unsqueeze(2))
        compound_inter_outputs = torch.multiply(compound_inter_outputs, compound_mask.unsqueeze(2))

        # Layer output
        protein_outputs = self.output(protein_inter_outputs, protein_inputs)
        compound_outputs = self.output(compound_inter_outputs, compound_inputs)
        
        protein_outputs = torch.multiply(protein_outputs, pocket_mask.unsqueeze(2))
        compound_outputs = torch.multiply(compound_outputs, compound_mask.unsqueeze(2))

        return protein_outputs, compound_outputs
        
    def forward(self, protein_features, pocket_attention_mask, compound_features, compound_attention_mask, pocket_mask, compound_mask):

        protein_att_outputs = protein_features
        compound_att_outputs = compound_features
        
        prots_att_outputs, comps_att_outputs = self.cross_att(protein_att_outputs, pocket_attention_mask, compound_att_outputs, compound_attention_mask, pocket_mask, compound_mask)
        prots_att_outputs, comps_att_outputs = self.self_att(protein_att_outputs, pocket_attention_mask, compound_att_outputs, compound_attention_mask, pocket_mask, compound_mask)

        protein_outputs, compound_outputs = self.output_fc(prots_att_outputs, comps_att_outputs, pocket_mask, compound_mask)
        
        return protein_outputs, pocket_attention_mask, compound_outputs, compound_attention_mask

class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, tmp_mask=None):

        output, att_probs = self.att(input_tensor, ctx_tensor, ctx_att_mask, tmp_mask)
     
        attention_output = self.output(output, input_tensor)
        attention_output = torch.multiply(attention_output, tmp_mask.unsqueeze(2))

        return attention_output, att_probs

class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask, tmp_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output, _ = self.self(input_tensor, input_tensor, attention_mask, tmp_mask) 
        attention_output = self.output(self_output, input_tensor) 
 
        attention_output = torch.multiply(attention_output, tmp_mask.unsqueeze(2))
        return attention_output

class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config["Architecture"]["hidden_size"], config["Architecture"]["hidden_size"]) 
        self.LayerNorm = BertLayerNorm(config["Architecture"]["hidden_size"], eps=1e-12) 
        self.dropout = nn.Dropout(config["Train"]["dropout"]) 

        if isinstance(config["Architecture"]["hidden_act"], str) or (sys.version_info[0] == 2 and isinstance(config["Architecture"]["hidden_act"], unicode)):
            self.transform_act_fn = ACT2FN[config["Architecture"]["hidden_act"]]
        else:
            self.transform_act_fn = config["Architecture"]["hidden_act"]

    def forward(self, hidden_states, input_tensor):
        
        hidden_states = self.dense(hidden_states) 

        hidden_states = self.dropout(hidden_states) 
        hidden_states = self.LayerNorm(hidden_states + input_tensor) 
        hidden_states = self.transform_act_fn(hidden_states)        
        
        return hidden_states
        
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config["Architecture"]["hidden_size"], config["Architecture"]["intermediate_size"]) 
        self.intermediate_act_fn = ACT2FN[config["Architecture"]["hidden_act"]]
        self.dropout = nn.Dropout(config["Train"]["dropout"])
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states) 
        hidden_states = self.intermediate_act_fn(hidden_states) 
        return hidden_states
        
class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config["Architecture"]["intermediate_size"], config["Architecture"]["hidden_size"]) 
        self.LayerNorm = BertLayerNorm(config["Architecture"]["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(config["Train"]["dropout"])
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states) 
        hidden_states = self.dropout(hidden_states) 
        hidden_states = self.LayerNorm(hidden_states + input_tensor) 
               
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.config = config
        if self.config["Architecture"]["hidden_size"] % self.config["Architecture"]["num_attention_heads"] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.config["Architecture"]["hidden_size"], self.config["Architecture"]["num_attention_heads"]))
        self.num_attention_heads = self.config["Architecture"]["num_attention_heads"] 
        self.attention_head_size = int(self.config["Architecture"]["hidden_size"] / self.config["Architecture"]["num_attention_heads"])
        self.all_head_size = self.config["Architecture"]["num_attention_heads"] * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = self.config["Architecture"]["hidden_size"]
        self.query = nn.Linear(self.config["Architecture"]["hidden_size"], self.all_head_size) 
        #self.query = nn.Linear(1024, self.all_head_size) 
        self.key = nn.Linear(ctx_dim, self.all_head_size) 
        self.value = nn.Linear(ctx_dim, self.all_head_size) 

        self.dropout = nn.Dropout(self.config["Train"]["dropout"])
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.config["Architecture"]["num_attention_heads"], self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None, tmp_att_mask = None):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context) 
        mixed_value_layer = self.value(context) 

        query_layer = self.transpose_for_scores(mixed_query_layer) 
        key_layer = self.transpose_for_scores(mixed_key_layer) 
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # key_layer.transpose(-1, -2): (Batch, Num_heads, Head_size, Max_lengths)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) 
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) 

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask 

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) 

        if tmp_att_mask is not None:
            attention_probs = torch.multiply(attention_probs, tmp_att_mask.unsqueeze(1).unsqueeze(3))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) 

        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs 