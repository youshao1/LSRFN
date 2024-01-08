import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv
# from sklearn.metrics.pairwise import cosine_similarity
from module.ShortAgg import ShortAggragation

def sce_loss(x, y, alpha):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss
def re_sce_loss(x, y, alpha):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = ((x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss
def cos_simi(arr1):
    similarity_matrix = torch.zeros((arr1.shape[0], arr1.shape[0]))
    for i, vector_A in enumerate(arr1):
        for j, vector_B in enumerate(arr1):
            similarity_matrix[i, j] = torch.cosine_similarity(vector_A.unsqueeze(0), vector_B.unsqueeze(0), dim=1)
    return similarity_matrix
def mask_schma(data,schma_mask_rate=0.5):
    mask_index_dict= HeteroData()
    for schma in data.schema_dict:
        tail = schma[1]
        num_nodes = data[tail].x.size(0)
        perm = torch.randperm(num_nodes)
        num_mask_nodes = int(schma_mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        # keep_nodes = perm[num_mask_nodes:]
        mask_index_dict[schma].masknodes=mask_nodes
    return  mask_index_dict
def noise(data):
    new_index = HeteroData()
    new_index["metapath_dict"] = data.metapath_dict
    for mp in data.metapath_dict:
        edge_index = data[mp].edge_index
        second_row = edge_index[1]
        shuffled_indices = torch.randperm(second_row.size(0))
        shuffled_second_row = second_row[shuffled_indices]
        new_index[mp].edge_index = torch.stack([edge_index[0], shuffled_second_row])
    return new_index
def noise_nei(data):
    new_nei_index = HeteroData()
    new_nei_index["schema_dict"] = data.schema_dict
    for mp in data.schema_dict:
        edge_index = data[mp].edge_index
        second_row = edge_index[1]
        shuffled_indices = torch.randperm(second_row.shape[0])
        shuffled_second_row = second_row[shuffled_indices]
        new_nei_index[mp].edge_index = torch.stack(
            [torch.from_numpy(edge_index[0]), torch.from_numpy(shuffled_second_row)])
    return new_nei_index
def mask(x, mask_rate=0.5, noise=0.05):
    num_nodes = x.size(0)
    perm = torch.randperm(num_nodes, device=x.device)
    num_mask_nodes = int(mask_rate * num_nodes)

    # random masking
    # num_mask_nodes = int(mask_rate * num_nodes)
    mask_nodes = perm[: num_mask_nodes]
    keep_nodes = perm[num_mask_nodes:]

    num_noise_nodes = int(noise * num_mask_nodes)
    # 长度为打乱1354
    perm_mask = torch.randperm(num_mask_nodes, device=x.device)
    # 在1354的基础上随机选择1258
    token_nodes = mask_nodes[perm_mask[: int((1 - noise) * num_mask_nodes)]]
    # 1354减去1258剩下的67个
    noise_nodes = mask_nodes[perm_mask[-int(noise * num_mask_nodes):]]
    # 2708中随机选择67个替换上面的67个
    noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

    return token_nodes, noise_nodes, noise_to_be_chosen, mask_nodes
class LSRFN(nn.Module):
    def __init__(
            self,
            use_data,
            hidden_dim,
            agg_method,
            attn_drop,
            feat_drop,
            alpha,
            dropout,
            metpath_mask_rate,
            nei_mask_rate,
            metapath_layers,
            heads,
            noise_rate,
            gat_drop
    ):
        super(LSRFN, self).__init__()
        self.hidden_dim = hidden_dim
        self.gat_drop=gat_drop
        self.heads = heads
        self.noise_rate=noise_rate
        self.metapath_layers=metapath_layers
        self.agg_method=agg_method
        self.metepath_mask_rate=metpath_mask_rate
        self.nei_mask_rate=nei_mask_rate
        self.dropout=dropout
        self.feat_drop = self.featdrop = nn.Dropout(feat_drop)
        self.HAN_layers = nn.ModuleList()
        for l in range(metapath_layers):
            self.HAN_layers.append(HANConv(hidden_dim,hidden_dim,use_data.metadata(),heads=heads,negative_slope=0.2,dropout=attn_drop))
        self.recon_adj = nn.Linear(hidden_dim,use_data[use_data.main_node].x.shape[0])
        self.fc = nn.ModuleDict({
            n_type: nn.Linear(
                use_data[n_type].x.shape[1],
                hidden_dim,
                bias=True
            )
            for n_type in use_data.use_nodes
        })
        self.enc_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.alpha = alpha
        self.localAggragation=ShortAggragation(agg_method,hidden_dim,use_data.schema_dict,gat_drop)
        self.reset_parameter()

        self.recon_features = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, use_data[use_data.main_node].x.shape[1])
        )

    def reset_parameter(self):
        for fc in self.fc.values():
            nn.init.xavier_normal_(fc.weight, gain=1.414)
    def forward(self, data,schmaData):
        h = {}
        h_schma= {}
        for i,n_type in enumerate(data.use_nodes):
            if i==0:
                h[n_type] = F.elu(
                    self.feat_drop(
                        self.fc[n_type](data[n_type].x)
                    )
                )
            else:
                h_schma[n_type] = F.elu(
                    self.feat_drop(
                        self.fc[n_type](data[n_type].x)
                    )
                )
        adj = torch.zeros(data[data.main_node].x.size(0)).to(torch.device("cuda:0"))
        for mp in data.metapath_dict:
            adj = adj + to_dense_adj(data[mp].edge_index).to(torch.device("cuda:0"))
        adj[adj > 1] = 1
        adj = adj.squeeze(0).to(torch.device("cuda:0"))
        token_nodes, noise_nodes, noise_to_be_chosen, mask_nodes = mask(h[data.main_node],self.metepath_mask_rate,self.noise_rate)
        h[data.main_node][token_nodes] = 0.0
        if self.noise_rate > 0:
            h[data.main_node][noise_nodes] = h[data.main_node][noise_to_be_chosen]
        else:
            h[data.main_node][noise_nodes] = 0.0
        h[data.main_node][token_nodes] += self.enc_mask_token
        schma_mask_dict = mask_schma(data,self.nei_mask_rate)
        for schma in schmaData.schema_dict:
            tail = schma[1]
            index = schma_mask_dict[schma].masknodes
            h_schma[tail][index]=0
        for i in range(self.metapath_layers):
            if i ==0:
                z_mp= self.HAN_layers[i](h,data.edge_index_dict)
            else:
                z_mp =  self.HAN_layers[i](z_mp,data.edge_index_dict)
        recon_adj1 = self.recon_adj(z_mp[data["main_node"]])
        loss_recon_adj = sce_loss(recon_adj1,adj,1)
        local_agg = self.localAggragation(h_schma,h,schmaData)
        all = z_mp[data["main_node"]]+local_agg
        recon_features = self.recon_features(all)
        loss_recon_fetures = sce_loss(recon_features[mask_nodes],data[data["main_node"]].x[mask_nodes],1)
        loss =loss_recon_fetures +loss_recon_adj
        return loss,loss_recon_adj,loss_recon_fetures

    def get_embeds(self, data,schmaData):
        h = {}
        h_schma = {}
        for i, n_type in enumerate(data.use_nodes):
            if i == 0:
                h[n_type] = F.elu(
                    self.feat_drop(
                        self.fc[n_type](data[n_type].x)
                    )
                )
            else:
                h_schma[n_type] = F.elu(
                    self.feat_drop(
                        self.fc[n_type](data[n_type].x)
                    )
                )
        for i in range(self.metapath_layers):
            if i == 0:
                z_mp = self.HAN_layers[i](h, data.edge_index_dict)
            else:
                z_mp = self.HAN_layers[i](z_mp, data.edge_index_dict)
        # 局部视图
        local_agg = self.localAggragation(h_schma,h,schmaData)
        all = z_mp[data["main_node"]]+local_agg
        return all.detach()