import torch
import argparse
import warnings
from utils.load import load_acm, load_dblp, load_freebase, load_imdb,load_aminer
from utils.evaluate import evaluate,evaluate_cluster
from module.lsrfn import LSRFN
from torch_geometric import seed_everything
import yaml
import numpy as np
# seed_everything(65536)
warnings.filterwarnings('ignore')


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "w" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


def train(args):
    if args.dataset == "Acm":
        load_data = load_acm
    elif args.dataset == "DBLP":
        load_data = load_dblp
    elif args.dataset == "Freebase":
        load_data = load_freebase
    elif args.dataset == "IMDB":
        load_data = load_imdb
    elif args.dataset == "Aminer":
        load_data = load_aminer
    # sum.mean,max
    data,schmaData = load_data()
    data =data.to(device)
    schmaData=schmaData.to(device)
    model = LSRFN(
        data,
        args.dim,
        args.method,
        args.attn_drop,
        args.feat_drop,
        args.alpha,
        args.dropout,
        args.long_mask_rate,
        args.short_mask_rate,
        args.metapath_layers,
        args.heads,
        args.noise_rate,
        args.gat_drop
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-05)

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        loss,loss_recon_adj,loss_reconfeatures= model(data,schmaData)
        print("epoch:{},loss:{},loss_reconfeatures:{},loss_recon_adj:{}".format(epoch + 1, loss.detach().cpu(),
                                                                                loss_reconfeatures.detach().cpu(),
                                                                                loss_recon_adj.detach().cpu()))
        loss.backward()

        optimizer.step()
    model.eval()
    embeds = model.get_embeds(data,schmaData)
    if args.task == 'classification':
        for ratio in args.ratio:
            evaluate(
                embeds,
                ratio,
                data[data.main_node][f'{ratio}_train_mask'],
                data[data.main_node][f'{ratio}_val_mask'],
                data[data.main_node][f'{ratio}_test_mask'],
                data[data.main_node].y,
                device,
                args.dataset,
                args.lr1,
                args.wd
            )
    elif args.task == 'clustering':
        label = data[data["main_node"]].y.cpu().data.numpy()
        nmi_list, ari_list = [], []
        embeds = embeds.cpu().data.numpy()
        for kmeans_random_state in range(10):
            nmi, ari = evaluate_cluster(embeds, label, label.max() + 1, kmeans_random_state)
            nmi_list.append(nmi)
            ari_list.append(ari)
        print("\t[clustering] nmi: [{:.4f}, {:.4f}] ari: [{:.4f}, {:.4f}]".format(np.mean(nmi_list), np.std(nmi_list),
                                                                                  np.mean(ari_list), np.std(ari_list)))


parser = argparse.ArgumentParser(description="LSRFN")
parser.add_argument("--dataset", type=str, default="Freebase")
# parser.add_argument("--task", type=str, default="classification")
parser.add_argument("--task", type=str, default="clustering")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--ratio", type=int, default=[20, 40, 60])
args = parser.parse_args()
if args.task=="classification":
    args = load_best_configs(args, "config.yaml")
else:
    args = load_best_configs(args, "clustering_config.yaml")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
train(args)
