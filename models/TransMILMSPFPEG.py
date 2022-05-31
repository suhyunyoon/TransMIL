import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


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
        out = self.attn(self.norm(x))
        
        x = x + out

        return x


class FPEG(nn.Module):
    def __init__(self, dim=512):
        super(FPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
        
        self.proj2 = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj2_1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2_2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
        
        self.proj3 = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj3_1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj3_2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        x_1 = self.proj(cnn_feat)+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x_1 = self.relu(x_1)
        
        x_2 = self.proj2(x_1)+self.proj2_1(x_1)+self.proj2_2(x_1)
        x_2 = self.relu(x_2)
        
        x = self.proj3(x_2)+self.proj3_1(x_2)+self.proj3_2(x_2)
        x = cnn_feat + x_1 + x_2 + x

        x = x.flatten(2).transpose(1, 2)
        #x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMILMSPFPEG(nn.Module):
    def __init__(self, n_classes):
        super(TransMILMSPFPEG, self).__init__()
        self.pos_layer = FPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())

        ## not use cls token 
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
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

        # #---->cls_token
        # B = h.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        # h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->FPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]
        
        h = self.norm(h)
        
        logits_all = self._fc2(h)
        Y_prob_all = F.softmax(logits_all, dim = 2)
        ind_token = int(torch.argmax(Y_prob_all[:,:,1], dim=1))
        logits = logits_all[:, ind_token]
        
        # #---->cls_token
        # h = h[:, 0]
                
        #---->predict
        # logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

if __name__ == "__main__":
    data = torch.randn((1, 6000, 512)).cuda()
    model = TransMILMSPFPEG(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)