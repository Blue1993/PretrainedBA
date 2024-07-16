from rdkit import Chem
from rdkit.Chem import AllChem

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

def get_mol_features(mol):
    atom_features_list, edges_list, edge_features_list = list(), list(), list()
    
    for atom in mol.GetAtoms():
        atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
        ]

        atom_features_list.append(atom_feature)
   
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()        
        j = bond.GetEndAtomIdx()
        edge_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
                allowable_features['possible_bond_dirs'].index(bond.GetBondDir())
            ]
        
        # add edges in both directions
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        edges_list.append((j, i))
        edge_features_list.append(edge_feature)
    
    # Graph connectivity in COO format with shape [2, num_edges]    
    edge_index = torch.tensor(edges_list, dtype=torch.long).T
    edge_features = torch.tensor(edge_features_list, dtype=torch.long)

    #return atom_features_list, eig_vecs, eig_vals, edge_index, edge_features, len(edges_list)
    return atom_features_list, edge_index, edge_features, len(edges_list)

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1
    
def remove_hydrogen(atom_features_list, edge_index, edge_features):
    atom_type = np.array(atom_features_list)[:, 0]

    nonH_index = np.array([True if a!= 0 else False for a in atom_type ])
    nonH_indexes = np.where(nonH_index == True)[0]

    nonH_atom_features_list = np.array(atom_features_list)[nonH_index]
    
    nonH_edge_index = list()

    for s, d in zip(edge_index.T[:, 0], edge_index.T[:, 1]):
        if int(s) not in nonH_indexes:
            nonH_edge_index.append(False)
        elif int(d) not in nonH_indexes:
            nonH_edge_index.append(False)
        else:
            nonH_edge_index.append(True)
            
    nonH_edge_index = np.array(nonH_edge_index)
    
    nonH_edge_features = edge_features[nonH_edge_index]
    nonH_edge_index = edge_index.T[nonH_edge_index]
    nonH_n_atoms, nonH_n_edges = nonH_atom_features_list.shape[0], nonH_edge_index.size()[0]
    
    return nonH_n_atoms, nonH_atom_features_list.tolist(), nonH_edge_features, nonH_edge_index.T, nonH_n_edges