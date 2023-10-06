import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils import *
from collections import defaultdict
from lib.models.layers import LocallyConnected2d
from lib.models.layers.feature_correction import *

class BidirectionalModel(nn.Module):
    """Take sequence of poses to predict: \n
    `walk_speed` : [ x, y, z ](normalized with leg length)\n
    `step_length`: [ l ] (does not differentiate the left/right)
    """
    def __init__(self, 
        seqlen, 
        # h_size=100,
        input_size=128,# per joint
        num_joints=24, 
        num_outputs=3,
        estime_phase=True,
        fc_size=40,
        num_layers=2,
        use_pareFeat=False,
        pare_concat=True,
        ):
        super().__init__()

        self.estim_phase = estime_phase
        multiple = 3
        self.num_outputs = num_outputs
        h_size = 300 if use_pareFeat else 100
        fc_size = 100 if use_pareFeat else 40
        self.num_layers = num_layers
        self.use_pareFeat = use_pareFeat
        self.pare_concat = pare_concat
        # if not self.use_pareFeat:
        #     self.h0 = nn.Parameter(torch.zeros(num_layers*2, 1, h_size).normal_(std=0.01), requires_grad=True)
        if not self.use_pareFeat:
            self.input_size = num_joints*multiple + 3
        else:
            self.input_size = input_size*num_joints
            self.dropout = nn.Dropout(0.2)
            cparam_out_channel = 30 if self.pare_concat else 128
            self.input_size += num_joints*cparam_out_channel if self.pare_concat else 0
            self.cparam_mpl = LocallyConnected2d(
                in_channels=3,
                out_channels=cparam_out_channel,
                output_size=[num_joints,1],
                kernel_size=1,
                stride=1,
            )

        self.rnn = nn.GRU(
            input_size=self.input_size, 
            hidden_size=h_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True,
        )
        if num_outputs>0:
            # split speed and step estimations 
            self.speed_mlp = nn.Sequential(
                nn.Linear(h_size*2*num_layers, fc_size),
                nn.LeakyReLU(0.05, inplace=True),
                nn.Linear(fc_size, 1)
            )
            self.step_mlp = nn.Sequential(
                nn.Linear(h_size*2*num_layers, fc_size),
                nn.LeakyReLU(0.05, inplace=True),
                nn.Linear(fc_size, 2)
            ) 

        if estime_phase:
            self.phase_mlp = nn.Sequential(
                nn.Linear(h_size*2, fc_size), # x2 for bidirectional
                nn.LeakyReLU(0.05, inplace=True),
                nn.Linear(fc_size, 4),
                nn.Tanh()
            )
    
    def forward(self, x, cparams=None):
        # ===== preprocess cparams with locallyConnectedLayer ===== #
        if self.use_pareFeat:
            assert cparams is not None
            b, n, cf = cparams.shape
            xc = self.cparam_mpl(cparams.reshape(b*n,cf,1,1).expand(b*n,cf,24,1)).reshape(b,n,-1)
            if self.pare_concat:
                # concatenate
                x = torch.cat([x, xc], dim=-1)
            else:
                # residual add
                x = x + self.dropout(xc)

        x, h = self.rnn(x,)
        h = h.permute(1,0,2)
        h = h.reshape(h.shape[0],-1)
        
        if self.num_outputs:
            y1 = self.speed_mlp(h)
            y2 = self.step_mlp(h)
            y = torch.cat((y1, y2), dim=-1)
        else:
            y = None
        
        if self.estim_phase:
            p = self.phase_mlp(x)
            return y, p, xc
        else:
            return y, None, xc

class _MultiAttention(nn.Module):
    "Multi-Attention class, hacked from KTD."
    def __init__(self, in_dim, encode_dim, out_dim, num_heads, num_joints=24, dropout=0.1,):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.encode_dim = encode_dim
        self.dim_head = encode_dim//num_heads
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(in_dim, encode_dim*3, bias=True)
        self.fc = nn.Linear(encode_dim, out_dim)
    
    def forward(self, x,):
        "Calculate temporal / spatial attn in parallel."
        b, n, _ = x.shape

        # apply FC to get Q, K, V
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.dim_head).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2] # each: [b, num_heads, n, dim_head]

        B, _, N, C = q.shape
        q = q.reshape(B, self.num_heads,N,C) #(B, num_heads, T, C)
        k = k.reshape(B, self.num_heads,N,C) #(B, num_heads, T, C)
        v = v.reshape(B, self.num_heads,N,C) #(B, num_heads, T, C)

        attn = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(C) # * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn) #(B, num_heads, T, T)

        x = torch.matmul(attn, v) #(B, num_heads, T, C)
        x = x.transpose(2,1).contiguous().reshape(B, N, C*self.num_heads)
        y = self.fc(x)
        y = self.dropout(y)

        return y

class AttentionBlock(nn.Module):
    def __init__(self,
        in_dim,
        encode_dim,
        out_dim,
        dropout=0.1,
        num_heads=1,
        ):
        super().__init__()
        self.in_dim = in_dim # will prepend a token
        self.norm1 = LayerNormalization(self.in_dim, eps=1e-8)
        self.mulattn = _MultiAttention(
            in_dim=self.in_dim,
            encode_dim=encode_dim,
            out_dim=self.in_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = LayerNormalization(self.in_dim, eps=1e-8)
        self.mlp = PositionwiseFeedForward(input_size=self.in_dim,#3072
                                            ff_size=encode_dim, #512
                                            output_size=encode_dim,
                                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x,):
        b,n,f = x.shape
        x_norm = self.norm1(x.reshape(b,n,-1))
        attn = self.mulattn(x_norm)
        attn = self.norm2(attn)
        x = x + self.dropout(attn)
        x = self.mlp(x)
        
        return self.dropout(x)

class AttentionModel(nn.Module):
    """
    Predict gait features (average speed and step_length, walk cycle signal)
    from pose features and camera parameters.
    """
    def __init__(
        self,
        x_size, # per joint
        encode_dim=512, # overall
        num_joints=24,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        ):
        super().__init__()
        self.x_size = x_size*num_joints
        self.input_size = max(self.x_size//4,encode_dim)
        self.num_joints = num_joints
        self.num_layers = num_layers
        # =====>> initialize network modules
        self.inp_mlp = LocallyConnected2d(
            in_channels=x_size,
            out_channels=self.input_size//num_joints,
            output_size=[num_joints,1],
            kernel_size=1,
            stride=1,
        )
        self.gf_token = nn.Parameter(torch.randn(1, 1, self.input_size))
        self.cparam_pe = LocomotivePE(d_feature=self.input_size, num_in=3,)
        self.pe = PositionalEncoding(d_model=self.input_size)
        self.in_dim = self.input_size
        self.encoder = nn.ModuleList([AttentionBlock(in_dim=self.in_dim if i==0 else encode_dim,
                                                     encode_dim=encode_dim,
                                                     out_dim=encode_dim,
                                                     num_heads=num_heads,
                                                     ) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.avg_mlp = nn.Sequential(
            nn.Linear(encode_dim, encode_dim//2),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(encode_dim//2,3),
        ) # [speed, lstep, rstep]
        self.cyc_mlp = nn.Sequential(
            nn.Linear(encode_dim, encode_dim//2),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(encode_dim//2,4),
            nn.Tanh(),
        ) # [left_cos, left_sin, right_cos, right_sin,] 

    def forward(self, x, cparams):
        # =====>> craft attention encoder output <<===== #
        b,n,_ = x.shape
        x = self.inp_mlp(x.reshape(b*n,-1,self.num_joints,1)).reshape(b,n,self.input_size)
        xc = self.cparam_pe(x, cparams)
        x_in = self.pe(xc)
        # concatenate avg token
        y = torch.cat([self.gf_token.expand(b,1,self.input_size),x_in], dim=1)
        # =====>> process with the encoder(s) <<===== #
        for i in range(self.num_layers):
            y = self.encoder[i](y)
            if i>0:
                y += y
        # =====>> regress average gait features & walk cycle <<===== #
        h = y[:,0,:] # b, 1, 512
        avg = self.avg_mlp(h)
        phase = self.cyc_mlp(y[:,1:,:])
        out_gf_token = y[:,:1,:]

        return avg, phase, xc, out_gf_token