# transformer encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from lib.models.layers import LocallyConnected2d


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(
            dim=-1,
            keepdim=True,
        )
        std = z.std(
            dim=-1,
            keepdim=True,
        )
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out


class PositionalEncoding(nn.Module):
    "PE function."
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # eventual pe will be cut from encoding of max_len
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            *  # arange for the range [0, dim_feature] with stride==2
            -(math.log(10000.0) / d_model))
        # torch.exp returns a new tensor with the exponential of the elements of the input tensor
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        return self.dropout(x)


class LocomotivePE(nn.Module):
    "Locomotive positional encoding based on gait features and cparams."
    def __init__(self,
                 d_feature,
                 num_in,
                 normalize=False,
                 dropout=0.1,
                 scale=1.,
                 use_fc=True):
        "If `normalize`, scale `feats` into [0,1]."
        super(LocomotivePE, self).__init__()
        self.d_feature = d_feature
        self.dropout = nn.Dropout(p=dropout)
        self.num_in = num_in
        self.scale = scale
        self.normalize = normalize
        self.use_fc = use_fc
        div_term = torch.exp(
            torch.arange(0, d_feature, 2) * -(math.log(10000) / d_feature))
        div_term = div_term.reshape(1, 1, d_feature // 2)
        self.fc_out = nn.Linear(self.num_in,
                                1)  # get a combined pe from wcyc & cparams
        # use 'register_buffer' to ensure that it's considered as part of the model
        self.register_buffer('div_term', div_term)

    def forward(self, x, feats,):
        """
        Generate positional encoding from normalized walk cycle `wcyc` __or__ camera parameters `cparams`.\n
        Add onto input `x`.
        """
        b, t, _ = x.shape
        if self.normalize:
            fmin = feats.min(1).values.unsqueeze(1)
            fmax = feats.max(1).values.unsqueeze(1)
            feats = (feats - fmin) / (fmax - fmin + 1e-9)
        if self.use_fc:
            feats = (self.fc_out(feats)).expand(
                b, t, self.d_feature // 2) * self.scale
        else:
            # =====>> Not using FC, to calculate euclidean distance
            feats = ((feats[:, :, 1:]**2) * feats[:, :, :1]).sum(-1) * self.scale
            feats = feats.unsqueeze(-1).expand(b, t, self.d_feature // 2)

        pe = torch.zeros((b, t, self.d_feature)).to(x.device)
        pe[:, :, 0::2] = torch.sin(
            feats * self.div_term.expand(b, t, self.d_feature // 2))
        pe[:, :, 1::2] = torch.cos(
            feats * self.div_term.expand(b, t, self.d_feature // 2))

        vis = False
        if vis:
            import numpy as np
            import matplotlib.pyplot as plt
            # rescale the value of jpe
            vals = pe[0].detach().cpu().numpy()
            vals -= vals.min()
            plt.imshow(vals, cmap='gray')
            # plt.savefig('test.png')

        x = x + pe

        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k) #4,90,1,d_feature

    if mask is not None:
        # mask = mask.unsqueeze(1).repeat(1, 4, 1, scores.shape[-1])
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # 4,90,1,1

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn  #


class Multi_Head_Attention(nn.Module):
    def __init__(
        self,
        input_size,  # 3792 / 3072
        dim_model,  # 1200 should be divisible by num_joints
        num_heads=2,
        seqlen=90,
        dropout=0.0,
    ):
        super(Multi_Head_Attention, self).__init__()

        # basic params
        self.num_heads = num_heads  # 2
        self.dim_head = dim_model // self.num_heads  # final feature = concatenated features of all head
        # weights to deduce Q,K,V
        self.fc_qkv = nn.Linear(input_size, dim_model*3)

        self.fc = nn.Linear(num_heads * self.dim_head, input_size)
        self.dropout = nn.Dropout(dropout)
        # LN for initial input
        self.layer_norm = LayerNormalization(d_hid=dim_model)

    def forward(self, x):
        b, n, f = x.shape
        qkv = self.fc_qkv(x).reshape(b,n,3,self.num_heads,f//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2,-1))*self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1,2).reshape(b,n,f)
        x = self.fc(x)
        x = self.dropout(x)

        return x  #b,seqlen,128*24


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """
    def __init__(self, input_size, ff_size, output_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super().__init__()
        #self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, output_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        #x_norm = self.layer_norm(x) # b, n, input_size
        return self.pwff_layer(x)

class JointWiseFeedForward(nn.Module):
    def __init__(self, input_size, ff_size, output_size, num_token, dropout=0.1):
        super().__init__()
        self.num_token = num_token
        self.dropout = nn.Dropout(dropout)
        self.jwff_layer1 = LocallyConnected2d(
            in_channels=input_size//num_token,
            out_channels=ff_size//num_token,
            output_size=[num_token, 1],
            kernel_size=1,
            stride=1,
        )
        self.jwff_layer2 = LocallyConnected2d(
            in_channels=ff_size//num_token,
            out_channels=output_size//num_token,
            output_size=[num_token, 1],
            kernel_size=1,
            stride=1,
        )
        self.actfun = nn.GELU()
        self.dropput = nn.Dropout(dropout)

    def forward(self, x):
        b,n,f = x.size()
        x = self.jwff_layer1(x.reshape(b*n,f//self.num_token, self.num_token, 1))
        x = self.dropout(self.actfun(x))
        x = self.jwff_layer2(x)
        x = self.dropout(x).squeeze(-1).reshape(b,n,-1)
        return x


class MultiAttention(nn.Module):
    "Multi-Attention class, hacked from MAED."
    def __init__(
        self,
        in_dim,
        encode_dim,
        out_dim,
        num_heads,
        num_token=24,
        use_spatial=False,
        spatial_lfc=False,
        dropout=0.1,
        split_mode='tasc',
        fake_concat=False, # for test without gait features
    ):
        """
        :params `num_token` number of tokens at one time step, for spatial attention encoding.\n
        :params `split_mode` whether use different input for temporal / spatial attention encoding.\n      
        """
        super().__init__()
        # =====>> basic parameters
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.encode_dim = encode_dim
        self.dim_head = encode_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_spatial = use_spatial
        self.spatial_lfc = spatial_lfc
        # =====>> construct sub-blocks
        self.qkv_t = nn.Linear(in_dim, encode_dim * 3, bias=True)
        # self.fc_out = nn.Linear(encode_dim, out_dim) # convert dimension
        if self.use_spatial:
            if split_mode.lower()=='tasc':
                in_dim_s = in_dim if fake_concat else in_dim + in_dim//num_token
                num_token += 0 if fake_concat else 1
            elif split_mode.lower()=='tcsa':
                in_dim_s = in_dim if fake_concat else in_dim - in_dim//num_token
                num_token -= 0 if fake_concat else 1
            elif split_mode.lower()=='leffws':
                in_dim_s = in_dim * num_token
            else:
                in_dim_s = in_dim
            self.ts_attn = nn.Linear(encode_dim * 2, encode_dim * 2)
            if self.spatial_lfc:
                # LFC may keep the feature dim to in_dim
                self.qkv_s = LocallyConnected2d(
                    in_channels=in_dim_s // num_token,
                    out_channels=3 * encode_dim // num_token,  # jointly regress qs,ks,vs
                    output_size=[num_token, 1],
                    kernel_size=1,
                    stride=1,
                )
            else:
                self.qkv_s = nn.Linear(in_dim_s, encode_dim * 3, bias=True)
            self.fc_s = nn.Linear(encode_dim, out_dim) # convert dimension
            # self.fc_out = nn.Linear(encode_dim, out_dim)
        self.fc_t = nn.Linear(encode_dim, out_dim) # convert dimension

    def forward(self, x, xs=None, smask=False,):
        "Calculate temporal / spatial attn in parallel."
        b, n, c, num_token = x.shape
        if xs is not None:
            _,_,_,n_tks = xs.shape
        else:
            n_tks = num_token
            xs = x
        # apply FC to get Q, K, V
        qkv_t = self.qkv_t(x.reshape(b, n, -1)).reshape(
            b, n, 3, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        qt, kt, vt = qkv_t[0], qkv_t[1], qkv_t[2]  # each: [b, num_heads, n, dim_head]
        x_t = self.forward_temporal(qt, kt, vt)  # [b,n,out_dim]
        if self.use_spatial:
            if self.spatial_lfc:
                qkv_s = self.qkv_s(xs.reshape(b * n, c, n_tks, 1)).reshape(
                    b, n, 3, self.num_heads, self.dim_head).permute(2, 0, 1, 3, 4)
            else:
                qkv_s = self.qkv_s(xs.reshape(b, n, -1)).reshape(
                    b, n, 3, self.num_heads, self.dim_head).permute(
                        2, 0, 1, 3, 4)  # (3,4,90,num_heads,dim_head//nj,25)
            qkv_s = qkv_s.reshape(3, b * n, self.num_heads, self.dim_head//n_tks, n_tks)
            qs, ks, vs = qkv_s[0], qkv_s[1], qkv_s[2]  # each: [b, seqlen, num_heads, dim_head//nj, nj]
            x_s = self.forward_spatial(qs, ks, vs)  # [b*n,num_heads]
            if smask:  # ignore the last token
                pass
            x_s = x_s.reshape(b, n, -1)
            # get the weigths for different attn
            alpha = torch.cat([x_t, x_s], dim=-1)
            alpha = alpha.mean(dim=1, keepdim=True)
            alpha = self.ts_attn(alpha).reshape(b, 1, -1, 2)  # [b,1,out_dim,2]
            alpha = alpha.softmax(dim=-1)
            # return the attention
            y = self.fc_t(x_t * alpha[:, :, :, 0]) + self.fc_s(x_s * alpha[:, :, :, 1])
            # y = self.fc_out(x_t * alpha[:, :, :, 0] + x_s * alpha[:, :, :, 1])
        else:
            y = self.fc_t(x_t)

        y = self.dropout(y)
        return y

    def forward_temporal(self,q,k,v,):
        B, _, N, C = q.shape
        qt = q.reshape(B, self.num_heads, N, C)  #(B, num_heads, T, C)
        kt = k.reshape(B, self.num_heads, N, C)  #(B, num_heads, T, C)
        vt = v.reshape(B, self.num_heads, N, C)  #(B, num_heads, T, C)

        attn = torch.matmul(qt, kt.transpose(-2, -1)) / math.sqrt(
            C)  # * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)  #(B, num_heads, T, T)

        x = torch.matmul(attn, vt)  #(B, num_heads, T, C)
        x = x.transpose(2, 1).contiguous().reshape(B, N, self.num_heads * C)
        return x

    def forward_spatial(self, q, k, v):
        N, num_heads, C, num_token = q.shape
        attn = (q.transpose(-2, -1) @ k)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)  # (N,num_heads,num_token,num_token)

        x = attn @ v.transpose(-2, -1)  # (N,num_heads,num_token,C)
        x = x.transpose(-1, -2).contiguous().reshape(N, num_heads,
                                                     C * num_token)
        return x

class TSAttnBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        encode_dim,
        out_dim,
        num_heads,
        num_token=24,
        use_spatial=False,
        spatial_lfc=False,
        dropout=0.1,
        split_mode=False,
        use_leff=False,
        use_jwff=False,
        #leff_in_dim=256, # per joint
        leff_h_size_mul=0.5,
        leff_smpl_feats=False,
        leff_fc_in=True, # apply linear proj or interpolation among channels
        fake_concat=False, # for test without gait features
    ):
        """
        :params `num_token` number of tokens at one time step, for spatial attention encoding.\n
        :params `split_mode` whether use different input for temporal / spatial attention encoding.\n      
        """
        super().__init__()
        if use_jwff: use_leff = False
        self.norm1 = LayerNormalization(in_dim, eps=1e-6)
        self.norm2 = LayerNormalization(in_dim, eps=1e-6)
        self.use_jwff = use_jwff
        self.use_leff = use_leff
        self.leff_smpl_feats = leff_smpl_feats
        if self.use_leff and self.leff_smpl_feats:
            num_token = 14**2+1
            self.mulattn = MultiAttention(
                in_dim=in_dim,
                encode_dim=512,
                out_dim=in_dim,
                num_heads=num_heads,
                num_token=num_token,
                dropout=dropout,
                use_spatial=use_spatial,
                spatial_lfc=False,
                split_mode='leffws',
                fake_concat=fake_concat,
            )
        else:
            self.mulattn = MultiAttention(
                in_dim=in_dim,
                encode_dim=encode_dim,
                out_dim=in_dim,
                num_heads=num_heads,
                num_token=num_token,
                dropout=dropout,
                use_spatial=use_spatial,
                spatial_lfc=spatial_lfc,
                split_mode=split_mode,
                fake_concat=fake_concat,
            )

            if self.use_leff:
                conv_in_dim = int(in_dim//num_token*leff_h_size_mul)
                self.ffn = LocallyEnhancedFeedForward(
                    in_features=in_dim//num_token,
                    conv_in_features=in_dim//num_token,
                    num_token=num_token,
                    hidden_features=conv_in_dim,
                    out_features=in_dim//num_token,
                    leff_fc_in=leff_fc_in,
                )
            elif self.use_jwff:
                self.ffn = JointWiseFeedForward(
                    input_size=in_dim,
                    ff_size=out_dim//2,
                    output_size=out_dim,
                    num_token=num_token,
                    dropout=dropout,
                )
            else:
                self.ffn = PositionwiseFeedForward(
                    input_size=in_dim,
                    ff_size=out_dim//2, #128*4
                    output_size=out_dim,
                    dropout=dropout)
                
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, xs=None,):
        assert x.dim() == 4
        # if self.use_leff and self.leff_smpl_feats:
        #     b,n,_,_ = x.shape
        #     attn = self.mulattn(x,)
        #     x = x.reshape(b,n,-1) + self.dropout(attn) # b*n, 14x14+1, 128
        # else:
        b, n, f, nj = x.shape
        attn = self.mulattn(x, xs=xs,)
        x = x.reshape(b, n, -1) + self.dropout(attn)  # double dropout, already one inside attn
        x = self.norm1(x)
        x_out = self.ffn(x)
        x = self.norm2(self.dropout(x_out) + x)
        return x  # [4, 90(91), 3072(3200)] | [b*n, 14x14+1, 128]
