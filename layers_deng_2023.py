#GNC layers used in Deng 2023
#copy from https://github.com/dengjianyuan/Respite_MPP/blob/main/src/models.py

"""A collection of molecular property prediction models"""
import torch
import torch.nn as nn
import torch.nn.functional as F

### for class RNN 
from dataset import inputTensor

### for class GCN/GIN
import math

from torch.nn import Parameter
from torch.nn import Linear, LayerNorm, ReLU

import torch_sparse
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_scatter import scatter
from torch_scatter import scatter_add

from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes


### recurrent neural networks (GRU)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=512, task_type='REG', num_layers=1, layer_type='GRU'):
        super(RNN, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.task_type = task_type
        self.num_layers = num_layers

        if layer_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers)

        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
  
        # initialize fc layer model weights using xavier_uniform
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
        nn.init.xavier_uniform_(self.fc3.weight.data)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    def forward(self, input, hidden):
        _, hidden_states = self.rnn(input, hidden)
        intermediate_output = torch.relu(self.fc1(hidden_states[0]))
        output = self.fc3(torch.relu(self.fc2(intermediate_output)))
        return output

# specify a set of atom and bond attributes for generating molecular graphs
num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3
num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

### graph convolutional networks implementation (reference: moclr-gcn_finetune.py)
def gcn_norm(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]

    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()
        self.emb_dim = emb_dim
        self.aggr = aggr
        self.weight = Parameter(torch.Tensor(emb_dim, emb_dim))
        self.bias = Parameter(torch.Tensor(emb_dim))
        self.reset_parameters()
        self.edge_embedding1 = nn.Embedding(num_bond_type, 1)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, 1)

        # initialize model weights for edge_embedding1 and edge_embedding2 
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr): 
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        
        # collect edge_embeddings
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # collect edge index
        edge_index, __ = gcn_norm(edge_index)

        # update x by multiply with weights
        x = x @ self.weight

        # forward propagate
        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings, size=None)

        # add bias to out
        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_attr):
        return x_j if edge_attr is None else edge_attr + x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

class GCN(nn.Module):
    def __init__(self, task='CLS', num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GCN, self).__init__()
        self.num_layer, self.emb_dim, self.feat_dim = num_layer, emb_dim, feat_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # specify two embedding layers for convert atom_type and chirality_tag into embeddings
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.drop_ratio = drop_ratio
        self.task = task

        # initialize a list of MLPs 
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GCNConv(emb_dim, aggr="add"))

        # initialize a list of BatchNorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        # specify pooling type
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')
        
        # specify a linear layer to convert hidden vectors (embeddings) into final features
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        # specify prediction heads: same prediction head for CLS or REG
        self.pred_head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim//2), nn.Softplus(), nn.Linear(self.feat_dim//2, 1)
        )

    def forward(self, batch_data):
        # specify x, edge_index, edge_attr
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr

        # get hidden vectors, i.e., embeddings
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        # update hidden vectors with the list of MLPs
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        # get final-layer hidden vectors (i.e., learned features) with a pooling layer and a linear layer
        h = self.pool(h, batch_data.batch) 
        h = self.feat_lin(h)

        # return final raw predictions 
        return self.pred_head(h)

### graph isomorphism networks implementation (reference: moclr-ginet_finetune.py)
class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), nn.ReLU(), nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
    
    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GIN(nn.Module):
    def __init__(self, task='CLS', num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean', pred_n_layer=2, pred_act='softplus'):
        super(GIN, self).__init__()
        self.num_layer, self.emb_dim, self.feat_dim, self.drop_ratio, self.task = num_layer, emb_dim, feat_dim, drop_ratio, task

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # initialize a list of MLPs 
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # initialize a list of BatchNorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        # specify pooling type
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')
        
        # specify a linear layer to convert hidden vectors (embeddings) into final features
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        # specify pred_n_layer
        self.pred_n_layer = max(1, pred_n_layer)
        # specify pred_head based on activation function
        if pred_act == 'relu':
            pred_head = [nn.Linear(self.feat_dim, self.feat_dim//2), nn.ReLU(inplace=True)]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([nn.Linear(self.feat_dim//2, self.feat_dim//2), nn.ReLU(inplace=True), ])
        elif pred_act == 'softplus':
            pred_head = [nn.Linear(self.feat_dim, self.feat_dim//2), nn.Softplus()]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([nn.Linear(self.feat_dim//2, self.feat_dim//2), nn.Softplus()])
        else:
            raise ValueError('Undefined activation function')
        # form pred_head
        pred_head.append(nn.Linear(self.feat_dim//2, 1))
        self.pred_head = nn.Sequential(*pred_head)

    def forward(self, batch_data):
        # specify x, edge_index, edge_attr
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr

        # get hidden vectors, i.e., embeddings
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        # update hidden vectors with the list of MLPs
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        # get final-layer hidden vectors (i.e., learned features) with a pooling layer and a linear layer
        h = self.pool(h, batch_data.batch)
        h = self.feat_lin(h)
        
        # return final raw predictions 
        return self.pred_head(h)
    