import numpy as np
import torch
from utils.logreg import LogReg
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score


##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################
def evaluate_cluster(embeds, y, n_labels, kmeans_random_state):
    Y_pred = KMeans(n_labels, random_state=kmeans_random_state).fit(embeds).predict(embeds)
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    return nmi, ari
def evaluate(embeds, ratio, train_mask, val_mask, test_mask, label, device, dataset, lr, wd, isTest=True):
    num_features = embeds.shape[1]
    num_classes = label.max() + 1
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[train_mask]
    val_embs = embeds[val_mask]
    test_embs = embeds[test_mask]

    train_lbls = label[train_mask]
    val_lbls = label[val_mask]
    test_lbls = label[test_mask]

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(50):
        log = LogReg(num_features, num_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []
        logits_list = []

        for _ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc

        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(
            roc_auc_score(
                y_true=F.one_hot(test_lbls).detach().cpu().numpy(),
                y_score=best_proba.detach().cpu().numpy(),
                multi_class='ovr'
            )
        )
    print("\t[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f} var: {:.4f}"
          .format(
            np.mean(macro_f1s),
            np.std(macro_f1s),
            np.mean(micro_f1s),
            np.std(micro_f1s),
            np.mean(auc_score_list),
            np.std(auc_score_list)
        )
    )
    return np.mean(macro_f1s),np.mean(micro_f1s),np.mean(auc_score_list)
    # if isTest:
    #     print("\t[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
    #           .format(
    #             np.mean(macro_f1s),
    #             np.std(macro_f1s),
    #             np.mean(micro_f1s),
    #             np.std(micro_f1s),
    #             np.mean(auc_score_list),
    #             np.std(auc_score_list)
    #         )
    #     )
    # else:
    #     return np.mean(macro_f1s_val), np.mean(macro_f1s)

    # f = open("result_"+dataset+str(ratio)+".txt", "a")
    # f.write(str(np.mean(macro_f1s))+"\t"+str(np.mean(micro_f1s))+"\t"+str(np.mean(auc_score_list))+"\n")
    # f.close()
