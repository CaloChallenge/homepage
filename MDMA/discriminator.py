import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim,dropout,avg_n=1601):
        super().__init__()
        self.avg_n=avg_n
        self.fc0 =  nn.Linear(embed_dim, hidden_dim)
        self.fc0_cls =  nn.Linear(embed_dim, hidden_dim)
        self.fc1 =  nn.Linear(hidden_dim+embed_dim, embed_dim)
        self.fc1_cls =  nn.Linear(hidden_dim+2, embed_dim)

        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.act = nn.LeakyReLU()
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, x_cls, mask,Einc):
        res = x.clone()
        res_cls = x_cls.clone()
        x = self.fc0(self.act(x))
        x_cls = self.ln(self.fc0_cls(self.act(x_cls)))
        x_cls = self.attn(x_cls, x, x, key_padding_mask=mask,need_weights=False)[0]
        x_cls = self.act(self.fc1_cls(torch.cat((x_cls,(~mask).float().sum(1).unsqueeze(1).unsqueeze(1)/self.avg_n,Einc.unsqueeze(1).unsqueeze(1)),dim=-1)))#+x.mean(dim=1).
        x = self.fc1(torch.cat((x,x_cls.expand(-1,x.shape[1],-1)),dim=-1))+res
        x_cls =x_cls+res_cls
        return x,x_cls


class Disc(nn.Module):
    def __init__(self, n_dim, latent_dim, hidden_dim,out_classes, num_layers, heads,dropout=0,avg_n=1601, **kwargs):
        """
        n_dim: dimension of the input (4, 1 for Energy, 3 for location)
        latent_dim: embedding dimension
        hidden_dim: hidden dimension used for attention agreggation
        out_classes: number of classes
        num_layers: number of layers
        heads: number of heads in the multihead attention
        dropout: dropout probability
        avg_n: average number of hits per dataset (1601 for dataset 2 and 4199 for dataset 3)
        """
        super().__init__()
        self.avg_n=avg_n
        self.embbed = nn.Linear(n_dim, latent_dim)
        self.embbed_cls = nn.Linear(latent_dim+1, latent_dim)
        self.encoder = nn.ModuleList([Block(embed_dim=latent_dim, num_heads=heads, hidden_dim=hidden_dim,dropout=dropout,avg_n=avg_n) for i in range(num_layers)])
        self.act = nn.LeakyReLU()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        self.out = nn.Linear(latent_dim, out_classes)
        self.softmax = nn.Softmax(dim=1)



    def forward(self, x, mask,Einc):
        """
        Input: Point Cloud of shape (B, N, 4)), mask of shape (B, N)
        Output: Probabilities of shape (B, out_classes)
        """
        x[mask]=0
        x = self.act(self.embbed(x))
        x_cls = (x.sum(1)/self.avg_n).unsqueeze(1).clone()
        x_cls= self.act(self.embbed_cls(torch.cat((x_cls,Einc.unsqueeze(1).unsqueeze(1)),dim=-1)))
        for layer in self.encoder:
            x,x_cls= layer(x, x_cls=x_cls, mask=mask, Einc=Einc)
        x_cls = self.act(self.ln(self.fc2(self.act(self.fc1(x_cls))))).squeeze(1)
        return self.softmax(self.out(x_cls))

if __name__ == "__main__":
    z = torch.randn(256, 40, 4,device="cuda")
    mask = torch.zeros((256, 40),device="cuda").bool()
    model = Disc(n_dim=4, latent_dim=16, hidden_dim=64,out_classes=3, num_layers=5, heads=16,dropout=0.2,avg_n=1601).cuda()
    s=model(z.cuda(),mask.cuda(),torch.ones((256),device="cuda"))
    print(s.std())