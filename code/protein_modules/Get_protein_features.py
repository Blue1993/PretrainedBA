import re
from transformers import BertModel, BertTokenizer

def get_protein_features(protein_seqs_dict):
    
    protein_features_dict = dict()
    
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case = False)
    prots_model = BertModel.from_pretrained("Rostlab/prot_bert") 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    prots_model = prots_model.to(device)
    prots_model = prots_model.eval()
    
    for PID in list(protein_seqs_dict.keys()):
        seqs_example = " ".join(list(re.sub(r"[UZOB]", "X", protein_seqs_dict[PID])))
        
        ids = tokenizer.batch_encode_plus([seqs_example], add_special_tokens = True, pad_to_max_length = True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device) 
        
        with torch.no_grad(): 
            embedding = prots_model(input_ids = input_ids, attention_mask = attention_mask)[0]
            embedding = embedding.cpu().numpy()
            seq_len = (attention_mask[0] == 1).sum()
    
            if seq_len < 1503:
                seq_emd = embedding[0][1:seq_len-1]            

            else:
                seq_len = 1502
                seq_emd = embedding[0][1:seq_len-1]
    
        protein_features_dict[PID] = seq_emd
    
    return protein_features_dict