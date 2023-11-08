import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        #h B C K L
        h = h.permute(0,2,1,3)
        Wh = torch.einsum('bnwl, wv->bnvl',(h, self.W)).contiguous()
        #Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        #print(e.shape, adj.shape)
        zero_vec = -9e15*torch.ones_like(e)
        #print(type(adj), adj)
        attention = torch.where(adj > 0, e, zero_vec).permute(0,2,3,1) #B N N L
        attention = F.softmax(attention, dim=2) #这里改成了dim = 2
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.einsum('bnnl, bnvl->bnvl',(attention, Wh)).contiguous().permute(0,2,1,3)
        #h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.einsum('bnvl, vi->bnil',(Wh, self.a[:self.out_features, :])).contiguous()
        Wh2 = torch.einsum('bnvl, vi->bnil',(Wh, self.a[self.out_features:, :])).contiguous()
        #Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        #Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.permute(0,2,1,3)
        #e = Wh1 + Wh2.T
        return self.leakyrelu(e).permute(0,3,1,2)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        #return F.log_softmax(x, dim=1)
        return x

class GATRNN_cell(nn.Module):
    def __init__(self, d_in, num_units, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GATRNN_cell, self).__init__()
        self.activation_fn = getattr(torch, activation)

        self.forget_gate = GAT(nfeat=d_in + num_units, nhid=num_units, nclass=num_units, dropout=0.5, alpha=0.2,
                                             nheads=8)
        self.update_gate = GAT(nfeat=d_in + num_units, nhid=num_units, nclass=num_units, dropout=0.5, alpha=0.2,
                                             nheads=8)
        self.c_gate = GAT(nfeat=d_in + num_units, nhid=num_units, nclass=num_units, dropout=0.5, alpha=0.2,
                                             nheads=8)

    def forward(self, x, h, adj):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        x_gates = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        x_c = torch.cat([x, r * h], dim=1)
        c = self.c_gate(x_c, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        return u * h + (1. - u) * c