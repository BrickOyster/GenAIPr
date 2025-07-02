import torch
from torch_geometric.data import Data

class SceneGraphBuilder:
    def __init__(self):
        self.object_vocab = {}
        self.relation_vocab = {}
        self.next_obj_idx = 0
        self.next_rel_idx = 0
        
    def build_graph(self, triples):
        nodes = []
        edges = []
        edge_attrs = []
        
        for triple in triples:
            # Add subject node
            if triple['subject'] not in self.object_vocab:
                self.object_vocab[triple['subject']] = self.next_obj_idx
                self.next_obj_idx += 1
            subj_idx = self.object_vocab[triple['subject']]
            
            # Add object node
            if triple['object'] not in self.object_vocab:
                self.object_vocab[triple['object']] = self.next_obj_idx
                self.next_obj_idx += 1
            obj_idx = self.object_vocab[triple['object']]
            
            # Add relation
            if triple['relation'] not in self.relation_vocab:
                self.relation_vocab[triple['relation']] = self.next_rel_idx
                self.next_rel_idx += 1
            rel_idx = self.relation_vocab[triple['relation']]
            
            # Add edges (bidirectional)
            edges.append((subj_idx, obj_idx))
            edge_attrs.append(rel_idx)
            edges.append((obj_idx, subj_idx))
            edge_attrs.append(rel_idx)
            
        edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
        
        return Data(x=None, edge_index=edge_index, edge_attr=edge_attr)