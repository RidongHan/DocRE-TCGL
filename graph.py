import math
import torch
import torch.nn as nn
from transformers import *
import torch.nn.functional as F


##### GAT #####
class GraphAttentionLayer(nn.Module):
    def __init__(self, edges, input_size, hidden_size, graph_drop, xavier_init=False):
        super(GraphAttentionLayer, self).__init__()
        self.edges = edges
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(size=(input_size, hidden_size)))
        self.W_edge = nn.ModuleList([nn.Linear(2*hidden_size, 1, bias=False) for i in (self.edges)])
        
        self.dropout = nn.Dropout(p=graph_drop, inplace=False)
        self.init_weight(xavier_init)

    def init_weight(self, xavier_init):
        gain = 1.414 if xavier_init else 1
        nn.init.xavier_uniform_(self.W.data, gain=gain)
        for m in self.W_edge:
            nn.init.xavier_uniform_(m.weight, gain=1.414)


    def forward(self, nodes_embed, node_adj_list):
        N = nodes_embed.shape[0]
        h = torch.matmul(nodes_embed, self.W)
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=-1)
        weight = torch.zeros(N*N).cuda()

        for idx, node_adj in enumerate(node_adj_list[:1]):
            weight += node_adj.view(N*N) * self.W_edge[idx](a_input).squeeze(dim=-1)

        weight = F.leaky_relu(weight, negative_slope=0.2).view(N, N)
        weight = weight.masked_fill(node_adj == 0, -9e15)
        attention = F.softmax(weight, dim=-1)
        attention = self.dropout(attention)
        dst = torch.matmul(attention, h)

        return dst


##### Multi-Head GAT #####
class GraphMultiHeadAttention(nn.Module):
    def __init__(self, edges, input_size, hidden_size, nhead=4, graph_drop=0.0, xavier_init=False):
        super(GraphMultiHeadAttention, self).__init__()
        self.nhead = nhead
        assert hidden_size % nhead == 0
        head_hidden = int(hidden_size/nhead)
        self.head_graph = nn.ModuleList([GraphAttentionLayer(edges, input_size, head_hidden, graph_drop, xavier_init=xavier_init) for _ in range(nhead)])
        self.dropout = nn.Dropout(p=graph_drop)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, nodes_embed, node_adj_list):
        x = []
        for cnt in range(0, self.nhead):
            x.append(self.head_graph[cnt](nodes_embed, node_adj_list))

        out = torch.cat(x, dim=-1)
        out = F.elu(out)
        self.layer_norm(self.dropout(out) + nodes_embed)
        return out
    

##### Graph Reason #####
class GraphReasonLayer(nn.Module):
    def __init__(self, edges, input_size, out_size, iters, graph_type="gat", graph_drop=0.0, graph_head=4, xavier_init=False):
        super(GraphReasonLayer, self).__init__()
        self.edges = edges
        self.iters = iters
        self.graph_type = graph_type
        
        if graph_type == "gat":
            self.block = nn.ModuleList(
                [GraphMultiHeadAttention(edges, input_size, out_size, nhead=graph_head, graph_drop=graph_drop, xavier_init=xavier_init) \
                 for i in range(iters)]
            )
        else:
            raise("[Error]: Graph Encoder choose error.")


    def forward(self, nodes_embed, node_adj):
        res = [nodes_embed]
        hi = nodes_embed
        for cnt in range(0, self.iters):
            hi = self.block[cnt](hi, node_adj)
            res.append(hi)
        
        out = res[-1]
        
        return out
