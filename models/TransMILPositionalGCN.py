import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from pygcn.layers import GraphConvolution


def dense_to_sparse(adj):
    # borrowed from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/sparse.py
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.
    Args:
        adj (Tensor): The dense adjacency matrix.
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])
    
    index = torch.stack(index, dim=0)
    #return torch.sparse.FloatTensor(index, edge_attr, adj.size())
    return torch.sparse_coo_tensor(index, edge_attr, adj.size(), device=adj.device)


class GCN(nn.Module):
    def __init__(self, dim, dropout, n_layers=1, skip_connection=False, n_skip=2):
        super(GCN, self).__init__()
        # add layers
        self.layers = []
        if n_layers > 0:
            layers = []
            for i in range(n_layers):
                layers.append(GraphConvolution(dim, dim))
            self.layers = nn.ModuleList(layers)

        self.dropout = dropout
        self.n_layers = n_layers
        self.skip_connection = skip_connection
        self.n_skip = n_skip

    def forward(self, x, adj):
        identity = []
        # forward hidden layers
        for layer in self.layers:
            # store previous identity
            if self.skip_connection:
                if len(identity) >= self.n_skip:
                    identity.pop(0)
                identity.append(x.clone())
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            # skip connection from previous 2
            for id_ in identity:
                x += id_

        return x


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PGCN(nn.Module):
    def __init__(self, dim=512, eps=1e-8, threshold=0.5):
        super(PGCN, self).__init__()
        self.gcn = GCN(dim, dropout=0.5, n_layers=3, skip_connection=False, n_skip=3)
        self.cos = nn.CosineSimilarity(dim=2)
        self.eps = eps
        self.relu = nn.ReLU()
        self.thres = threshold

    def forward(self, x):
        #B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:] #[B(1), (1, N-1), 512]

        # Generate adjacency matrix
        # with torch.no_grad():
        feat_n = feat_token.norm(dim=2)[:,:,None]
        feat_norm = feat_token / torch.max(feat_n, self.eps * torch.ones_like(feat_n))
        sim = torch.bmm(feat_norm, feat_norm.transpose(1,2))
        #sim = self.cos(feat_token, feat_token.transpose(1,2), eps=self.eps)
        
        adj = sim # >= self.thres
        #print(sim.detach().cpu(), adj.size(), adj.size(1)*adj.size(2), adj.sum())

        # Pruning
        topk = max(int(adj.size(1) * 0.01), 100)
        topk_val, _ = torch.topk(adj, topk, largest=False, sorted=True, dim=-1)
        # get topk value scalar
        topk_val = topk_val[:,:,-1:]
        # Remove weights of useless edges
        adj = torch.where(adj >= topk_val[:,:,-1:], adj, torch.zeros_like(adj)).float()

        # iteration by Mini-Batch
        f_arr = []
        for i, f in enumerate(feat_token):
            adj_i = dense_to_sparse(adj[i])
            f_arr.append(self.gcn(f, adj_i))

        x = torch.cat((cls_token.unsqueeze(1), torch.stack(f_arr, dim=0)), dim=1)
        return x


class TransMILPositionalGCN(nn.Module):
    def __init__(self, n_classes, input_dim=1024):
        super(TransMILPositionalGCN, self).__init__()
        self.pos_layer = PGCN(dim=512) #PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU()) # 원래는 1024, 512 
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)


    def forward(self, **kwargs):

        h = kwargs['data'].float() #[B, n, 1024]
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.pos_layer(h) # ???
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        #h = self.pos_layer(h, _H, _W) #[B, N, 512]
        h = self.pos_layer(h) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict


if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMILPositionalGCN(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
# if __name__ == "__main__":
#     adj = torch.randn(4, 4)
#     adj -= adj.min()
#     adj /= adj.max()
#     adj = adj ** 3

#     pack = dense_to_sparse(adj)
#     import pdb; pdb.set_trace()
