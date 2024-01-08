import torch
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv, Sequential, BatchNorm, GATConv
from torch_geometric.utils import scatter
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
from module.aggregate import  Aggregator
class GAT(MessagePassing):
    def __init__(self, in_channels, dropout, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.dropout = dropout

        self.att_src = Parameter(torch.Tensor(1, in_channels))
        self.att_dst = Parameter(torch.Tensor(1, in_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, edge_index):
        x_src, x_dst = x

        alpha_src = (x_src * self.att_src).sum(-1)
        alpha_dst = (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class ShortAggragation(nn.Module):
    def __init__(self,method,hidden_dim,schema_dict,gat_drop):
        super(ShortAggragation, self).__init__()
        self.method = method
        self.project = nn.ModuleDict({
            n_type[1]: nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=True
            )
            for n_type in schema_dict
        })
        self.fc = nn.Linear(hidden_dim*2, 1, bias=False)
        self.relu = nn.Sigmoid()
        self.schema_dict = {s: i for i, s in enumerate(schema_dict)}
        self.intra = nn.ModuleList([
            GAT(hidden_dim, gat_drop)
            for _ in range(len(self.schema_dict))
        ])
        self.inter = Aggregator(hidden_dim, gat_drop)

    def threshold(self,x):
        return torch.where(x >1, torch.tensor(1.),x)

    def forward(self, h_schma, h, data):
        if self.method!='gat':
            i = 0
            for mp in data.schema_dict:
                tar = mp[0]
                if i == 0:
                    tail = mp[1]
                    edge_index = torch.tensor(data[mp].edge_index).to(torch.device("cuda:0"))
                    x = self.project[tail](h_schma[tail])
                    x = x[edge_index[0]]
                    out = scatter(x, edge_index[1], dim=0, dim_size=h[tar].size(0), reduce=self.method)
                    i += 1
                else:
                    tail = mp[1]
                    edge_index = torch.tensor(data[mp].edge_index).to(torch.device("cuda:0"))
                    x = self.project[tail](h_schma[tail])
                    x = x[edge_index[0]]
                    out = scatter(x, edge_index[1], dim=0, dim_size=h[tar].size(0), reduce=self.method) + out
        else:
            embeds = []
            for mp in data.schema_dict:
                tar = mp[0]
                tail=mp[1]
                h1 = self.project[tail](h_schma[tail])
                x = h1, h[tar]
                embed = self.intra[self.schema_dict[mp]](x, data[mp].edge_index)
                embeds.append(embed)
            embeds = torch.stack(embeds)
            out=self.inter(embeds)
        return out