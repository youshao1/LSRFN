import torch
import torch.nn as nn


class Aggregator(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Aggregator, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.att = nn.Parameter(torch.Tensor(1, hidden_dim))

        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
    
    def forward(self, embeds):
        sp = torch.tanh(self.fc(embeds)).mean(1)

        beta = self.attn_drop(self.att) @ sp.T
        beta = torch.softmax(beta, dim=1).view(-1, 1, 1)

        z = (beta * embeds).sum(0)
        return z