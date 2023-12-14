import torch
import os
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T

class GCN_chess(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim,hidden_dim))
        for i in range(1, num_layers):
          if i < num_layers - 1:
            self.convs.append(GCNConv(hidden_dim,hidden_dim))
          elif i == num_layers - 1:
            self.convs.append(GCNConv(hidden_dim,output_dim))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for i in range(1, num_layers-1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.softmax = torch.nn.LogSoftmax(-1)

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        out = x
        num_layers_from_convs = len(self.convs)
        for i in range(num_layers_from_convs):
          out = self.convs[i](out,edge_index=adj_t)

          if i < num_layers_from_convs-1:
            out = self.bns[i](out)
            out = torch.nn.functional.relu(out)
            out = F.dropout(out, p = self.dropout, training = self.training)

          elif i < num_layers_from_convs and not (self.return_embeds) : 
            out = self.softmax(out)
          elif i >= num_layers_from_convs:
            print("error: you somehow skipped past the number of layers!\n") 

        return out
