import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class PositionalEncoding(nn.Module):
    
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 30000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
        # self.fc = nn.Linear(dim,dim)
        # self.fc1 = nn.Linear(dim, dim)
        # self.fc2 = nn.Linear(dim, dim)
        # self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]

        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # x_ = self.relu(self.fc(feat_token))
        # x_ = self.relu(self.fc1(x_))
        # x_ = self.fc2(x_) + feat_token

        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMILPositionalEncoding(nn.Module):
    def __init__(self, n_classes, input_dim=1024):
        super(TransMILPositionalEncoding, self).__init__()
        self.pos_layer = PositionalEncoding(dim=512) #PPEG(dim=512)
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
        #h = self.pos_layer(h) # ???
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
    model = TransMILPositionalEncoding(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
