import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import HeteroData
from torch_geometric.datasets import IMDB, AMiner
from torch_geometric.transforms import AddMetaPaths


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None,
                     test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size,
        val_size, test_size)

    # print('number of training: {}'.format(len(train_indices)))
    # print('number of validation: {}'.format(len(val_indices)))
    # print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['val'] = val_mask
    mask['test'] = test_mask
    return mask

def preprocess_sp_features(features):
    features = features.tocoo()
    row = torch.from_numpy(features.row)
    col = torch.from_numpy(features.col)
    e = torch.stack((row, col))
    v = torch.from_numpy(features.data)
    x = torch.sparse_coo_tensor(e, v, features.shape).to_dense()
    x.div_(x.sum(dim=-1, keepdim=True).clamp_(min=1.))
    return x


def preprocess_th_features(features):
    x = features.to_dense()
    x.div_(x.sum(dim=-1, keepdim=True).clamp_(min=1.))
    return x


def nei_to_edge_index(nei, reverse=False):
    edge_indexes = []

    for src, dst in enumerate(nei):
        src = torch.tensor([src], dtype=dst.dtype, device=dst.device)
        src = src.repeat(dst.shape[0])
        if reverse:
            edge_index = torch.stack((dst, src))
        else:
            edge_index = torch.stack((src, dst))

        edge_indexes.append(edge_index)

    return torch.cat(edge_indexes, dim=1)


def sp_feat_to_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sp_adj_to_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return indices


def make_sparse_eye(N):
    e = torch.arange(N, dtype=torch.long)
    e = torch.stack([e, e])
    o = torch.ones(N, dtype=torch.float32)
    return torch.sparse_coo_tensor(e, o, size=(N, N))


def make_sparse_tensor(x):
    row, col = torch.where(x == 1)
    e = torch.stack([row, col])
    o = torch.ones(e.shape[1], dtype=torch.float32)
    return torch.sparse_coo_tensor(e, o, size=x.shape)


def load_dblp():
    path = "./data/dblp/"
    ratio = [20, 40, 60]

    label = np.load(path + "labels.npy").astype('int32')
    # nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.load_npz(path + "p_feat.npz").astype("float32")
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    # pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.LongTensor(label)
    # nei_p = nei_to_edge_index([torch.LongTensor(i) for i in nei_p], True)
    feat_p = preprocess_sp_features(feat_p)
    feat_a = preprocess_sp_features(feat_a)
    apa = sp_adj_to_tensor(apa)
    apcpa = sp_adj_to_tensor(apcpa)
    aptpa = sp_adj_to_tensor(aptpa)
    ap = np.genfromtxt(path + "pa.txt").astype('int64').T
    copy=ap[1].copy()
    ap[1]=ap[0]
    ap[0]=copy
    # pos = sp_adj_to_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    schemaData = HeteroData()
    mask = torch.tensor([False] * feat_a.shape[0])
    data['a'].x = feat_a
    data['a'].y = label

    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['a'][train_mask_l] = train_mask
        data['a'][val_mask_l] = val_mask
        data['a'][test_mask_l] = test_mask

    data['p'].x = feat_p
    schemaData[('a', 'p')].edge_index = torch.tensor(ap)
    schemaData[('a', 'p')].edge_index[[0, 1]] = schemaData[('a', 'p')].edge_index[[1, 0]]
    data[('a', 'p', 'a')].edge_index = apa
    data[('a', 'pcp', 'a')].edge_index = apcpa
    data[('a', 'ptp', 'a')].edge_index = aptpa
    # data[('a', 'pos', 'a')].edge_index = pos
    data["dataset"]="dblp"
    metapath_dict = {
        ('a', 'p', 'a'): None,
        ('a', 'pcp', 'a'): None,
        ('a', 'ptp', 'a'): None
    }

    schema_dict = {
        ('a', 'p'): None
    }

    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    schemaData['schema_dict'] = schema_dict
    data['main_node'] = 'a'
    data['use_nodes'] = ('a', 'p')

    return data,schemaData


def load_acm():
    path = "./data/acm/"
    ratio = [20, 40, 60]
    label = np.load(path + "labels.npy").astype('int32')
    # nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    # nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    feat_p = sp.load_npz(path + "p_feat.npz").astype("float32")
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_s = make_sparse_eye(60)
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    # pos = sp.load_npz(path + "pos.npz")

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.LongTensor(label)
    # nei_a = nei_to_edge_index([torch.LongTensor(i) for i in nei_a])
    # nei_s = nei_to_edge_index([torch.LongTensor(i) for i in nei_s])
    feat_p = preprocess_sp_features(feat_p)
    feat_a = preprocess_sp_features(feat_a)
    feat_s = preprocess_th_features(feat_s)
    pap = sp_adj_to_tensor(pap)
    psp = sp_adj_to_tensor(psp)
    # pos = sp_adj_to_tensor(pos)
    pa = np.genfromtxt(path + "pa.txt").astype('int64').T

    ps = np.genfromtxt(path + "ps.txt").astype('int64').T


    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    schemaData = HeteroData()
    mask = torch.tensor([False] * feat_p.shape[0])

    data['p'].x = feat_p
    data['a'].x = feat_a
    data['s'].x = feat_s
    data['p'].y = label

    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['p'][train_mask_l] = train_mask
        data['p'][val_mask_l] = val_mask
        data['p'][test_mask_l] = test_mask

    schemaData[('p', 'a')].edge_index = torch.tensor(pa)
    schemaData[('p', 'a')].edge_index[[0, 1]] = schemaData[('p', 'a')].edge_index[[1, 0]]
    schemaData[('p', 's')].edge_index = torch.tensor(ps)
    schemaData[('p', 's')].edge_index[[0, 1]] = schemaData[('p', 's')].edge_index[[1, 0]]
    data[('p', 'a', 'p')].edge_index = pap
    data[('p', 's', 'p')].edge_index = psp
    # data[('p', 'pos', 'p')].edge_index = pos

    metapath_dict = {
        ('p', 'a', 'p'): None,
        ('p', 's', 'p'): None
    }

    schema_dict = {
        ('p', 'a'): None,
        ('p', 's'): None
    }
    # schema_dict = {
    #     ('p', 'a'): None,
    #     ('p', 's'): None
    # }

    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    schemaData['schema_dict'] = schema_dict
    data['main_node'] = 'p'
    data['use_nodes'] = ('p', 'a', 's')
    # data['use_nodes'] = ('p')
    data["dataset"] = "acm"
    return data,schemaData


def load_aminer():
    ratio = [20, 40, 60]
    path = "./data/aminer/"

    label = np.load(path + "labels.npy").astype('int32')
    # nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    # nei_r = np.load(path + "nei_r.npy", allow_pickle=True)
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = make_sparse_eye(6564)
    feat_a = make_sparse_eye(13329)
    feat_r = make_sparse_eye(35890)
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pa = np.genfromtxt(path + "pa.txt").astype('int64').T

    pr = np.genfromtxt(path + "pr.txt").astype('int64').T

    # mw = np.genfromtxt(path + "mw.txt").astype('int64').T
    # pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.LongTensor(label)
    # nei_a = nei_to_edge_index([torch.LongTensor(i) for i in nei_a])
    # nei_r = nei_to_edge_index([torch.LongTensor(i) for i in nei_r])
    feat_p = preprocess_th_features(feat_p)
    feat_a = preprocess_th_features(feat_a)
    feat_r = preprocess_th_features(feat_r)
    pap = sp_adj_to_tensor(pap)
    prp = sp_adj_to_tensor(prp)
    # pos = sp_adj_to_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    schemaData = HeteroData()
    mask = torch.tensor([False] * feat_p.shape[0])

    data['p'].x = feat_p
    data['a'].x = feat_a
    data['r'].x = feat_r
    data['p'].y = label

    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['p'][train_mask_l] = train_mask
        data['p'][val_mask_l] = val_mask
        data['p'][test_mask_l] = test_mask

    schemaData[('p', 'a')].edge_index = torch.tensor(pa)
    schemaData[('p', 'a')].edge_index[[0, 1]] = schemaData[('p', 'a')].edge_index[[1, 0]]
    schemaData[('p', 'r')].edge_index = torch.tensor(pr)
    schemaData[('p', 'r')].edge_index[[0, 1]] = schemaData[('p', 'r')].edge_index[[1, 0]]
    data[('p', 'a', 'p')].edge_index = pap
    data[('p', 'r', 'p')].edge_index = prp
    # data[('p', 'pos', 'p')].edge_index = pos

    metapath_dict = {
        ('p', 'a', 'p'): None,
        ('p', 'r', 'p'): None
    }

    schema_dict = {
        ('p', 'a'): None,
        ('p', 'r'): None
    }

    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    schemaData['schema_dict'] = schema_dict
    data['main_node'] = 'p'
    data['use_nodes'] = ('p', 'a', 'r')
    data["dataset"] = "aminer"
    return data,schemaData


def load_freebase():
    ratio = [20, 40, 60]
    path = "./data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    # nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    # nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    # nei_w = np.load(path + "nei_w.npy", allow_pickle=True)

    feat_m = make_sparse_eye(3492)
    feat_d = make_sparse_eye(2502)
    feat_a = make_sparse_eye(33401)
    feat_w = make_sparse_eye(4459)

    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    ma = np.genfromtxt(path + "ma.txt").astype('int64').T
    md = np.genfromtxt(path + "md.txt").astype('int64').T
    mw = np.genfromtxt(path + "mw.txt").astype('int64').T



    # pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.LongTensor(label)
    # nei_d = nei_to_edge_index([torch.LongTensor(i) for i in nei_d])
    # nei_a = nei_to_edge_index([torch.LongTensor(i) for i in nei_a])
    # nei_w = nei_to_edge_index([torch.LongTensor(i) for i in nei_w])

    feat_m = preprocess_th_features(feat_m)
    feat_d = preprocess_th_features(feat_d)
    feat_a = preprocess_th_features(feat_a)
    feat_w = preprocess_th_features(feat_w)

    mam = sp_adj_to_tensor(mam)
    mdm = sp_adj_to_tensor(mdm)
    mwm = sp_adj_to_tensor(mwm)
    # pos = sp_adj_to_tensor(pos)

    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    schmaData = HeteroData()
    mask = torch.tensor([False] * feat_m.shape[0])

    data['m'].x = feat_m
    data['d'].x = feat_d
    data['a'].x = feat_a
    data['w'].x = feat_w
    data['m'].y = label

    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['m'][train_mask_l] = train_mask
        data['m'][val_mask_l] = val_mask
        data['m'][test_mask_l] = test_mask

    #data[('d', 'm')].edge_index = nei_d.flip([0])
    #data[('a', 'm')].edge_index = nei_a.flip([0])
    #data[('w', 'm')].edge_index = nei_w.flip([0])
    data[('m', 'a', 'm')].edge_index = mam
    data[('m', 'd', 'm')].edge_index = mdm
    data[('m', 'w', 'm')].edge_index = mwm
    schmaData[('m', 'a')].edge_index = torch.tensor(ma)
    schmaData[('m', 'a')].edge_index[[0,1]]=schmaData[('m', 'a')].edge_index[[1,0]]
    schmaData[('m', 'd')].edge_index = torch.tensor(md)
    schmaData[('m', 'd')].edge_index[[0, 1]] = schmaData[('m', 'd')].edge_index[[1, 0]]
    schmaData[('m', 'w')].edge_index = torch.tensor(mw)
    schmaData[('m', 'w')].edge_index[[0, 1]] = schmaData[('m', 'w')].edge_index[[1, 0]]
    num_main_nodes = feat_m.shape[0]
    # data[('m', 'pos', 'm')].edge_index = pos

    metapath_dict = {
        ('m', 'a', 'm'): None,
        ('m', 'd', 'm'): None,
        ('m', 'w', 'm'): None
    }


    schema_dict = {
        ('m', 'a'): None,
        ('m', 'd'): None,
        ('m', 'w'): None
    }

    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    schmaData['schema_dict'] = schema_dict
    data['main_node'] = 'm'
    data['use_nodes'] = ('m', 'a', 'd', 'w')
    #data['use_nodes'] = ('m')

    data["dataset"] = "freebase"
    return data,schmaData


def load_imdb():
    ratio = [20, 40, 60]
    data = IMDB(root='./data/imdb/')[0]
    metapaths = [[("movie", "director"), ("director", "movie")],
                 [("movie", "actor"), ("actor", "movie")]]

    schmaData = HeteroData()
    schmaData["movie","director"].edge_index=data["movie","director"].edge_index
    # schmaData["movie","director"].edge_index[[0, 1]] = schmaData["movie","director"].edge_index[[1, 0]]
    schmaData["movie", "actor"].edge_index = data["movie", "actor"].edge_index
    # schmaData["movie", "actor"].edge_index[[0, 1]] = schmaData["movie", "actor"].edge_index[[1, 0]]
    data = AddMetaPaths(metapaths, drop_orig_edges=True)(data)
    schmaData["movie", "director"].edge_index[[0, 1]] = schmaData["movie", "director"].edge_index[[1, 0]]
    schmaData["movie", "actor"].edge_index[[0, 1]] = schmaData["movie", "actor"].edge_index[[1, 0]]
    metapath_dict = {
        ('movie', 'metapath_0', 'movie'): None,
        ('movie', 'metapath_1', 'movie'): None
    }

    schema_dict = {
        ('movie', 'actor'): None,
        ('movie', 'director'): None
    }

    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    schmaData['schema_dict'] = schema_dict
    data['main_node'] = 'movie'
    data['use_nodes'] = ('movie', 'actor', 'director')
    #data['use_nodes'] = ('movie',)

    for r in ratio:
        mask = train_test_split(
            data['movie'].y.detach().cpu().numpy(), seed=np.random.randint(0, 35456, size=1), train_examples_per_class=r,
            val_size=1000, test_size=None)
        train_mask_l = f"{r}_train_mask"
        train_mask = mask['train'].astype(bool)
        val_mask_l = f"{r}_val_mask"
        val_mask = mask['val'].astype(bool)

        test_mask_l = f"{r}_test_mask"
        test_mask = mask['test'].astype(bool)

        data['movie'][train_mask_l] = train_mask
        data['movie'][val_mask_l] = val_mask
        data['movie'][test_mask_l] = test_mask
    data["dataset"] = "imdb"
    return data,schmaData

def load_aminer_pyg():
    ratio = [20, 40, 60]
    data = AMiner(root='./data/aminer_pyg/')[0]
    metapaths = [[("author", "paper"), ("paper", "author")],
                 [("author", "paper"), ("paper", "venue"), ("venue", "paper"), ("paper", "author")]]
    data = AddMetaPaths(metapaths, drop_orig_edges=True)(data)

    metapath_dict = {
        ('author', 'metapath_0', 'author'): None,
        ('author', 'metapath_1', 'author'): None
    }

    schema_dict = {
        ('venue', 'paper'): None,
        ('venue', 'author'): None
    }

    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    data['main_node'] = 'paper'
    data['use_nodes'] = ('paper', 'venue', 'author')
    # pos = sp.load_npz("./pos.npz")
    # pos = sp_adj_to_tensor(pos)
    # data[('movie', 'pos', 'movie')].edge_index = pos
    for r in ratio:
        mask = train_test_split(
            data['movie'].y.detach().cpu().numpy(), seed=np.random.randint(0, 35456, size=1),
            train_examples_per_class=r,
            val_size=1000, test_size=None)
        train_mask_l = f"{r}_train_mask"
        train_mask = mask['train'].astype(bool)
        val_mask_l = f"{r}_val_mask"
        val_mask = mask['val'].astype(bool)

        test_mask_l = f"{r}_test_mask"
        test_mask = mask['test'].astype(bool)

        data['movie'][train_mask_l] = train_mask
        data['movie'][val_mask_l] = val_mask
        data['movie'][test_mask_l] = test_mask
    data["dataset"] = "imdb"
    return data
