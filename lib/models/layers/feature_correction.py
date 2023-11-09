import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from typing import NewType, Optional
from dataclasses import dataclass, asdict, fields
from timm.models.layers import trunc_normal_
from lib.models.layers import LocallyConnected2d
from lib.models.layers.attention_utils import LocomotivePE, TSAttnBlock, PositionalEncoding
from lib.models.layers.gait_feat_encoder import BidirectionalModel

fun_normalize = lambda x: x/(torch.norm(x,dim=-1,keepdim=True)+1e-12)

Tensor = NewType('Tensor', torch.Tensor)

class FeatCorrector(nn.Module):
    use_gt_gaitfeat = True # use GT P_G to supervise GaitFeat-Net
    def __init__(self,
        x_size, # per joint
        gfeat_out_channel=128, # dim of extra token
        num_avg_gfeat=3, # number of averaged gait features
        estim_phase=True,
        seqlen=90,
        num_layers=2,
        h_size=1024, # TE encode_dim
        num_joints=24,
        num_transformer_head=1,
        attn_embed_size=512,
        dropout=0.1,
        use_jwff=False,
        ):
        "Pose Feature correction based on estimated gait features. Always use residual."
        assert gfeat_out_channel%x_size==0
        super().__init__()
        self.num_avg_gfeat = num_avg_gfeat
        self.num_joints = num_joints
        # ===== GaitFeat module ===== #
        self.tencode_type = temporal_encode.lower()
        self.num_layers = num_layers
        self.unit_size = x_size 
        self.x_size = x_size*num_joints 
        self.featnet = BidirectionalModel(
            seqlen=seqlen,
            input_size=x_size, # per joint
            num_outputs=num_avg_gfeat,
            estime_phase=estim_phase,
            num_joints=num_joints,
            use_pareFeat=True,
        )
        # =====>> Feature Correction (per frame) <<===== #
        self.gf_mode = gf_mode.lower()
        self.input_size = self.x_size + gfeat_out_channel
        self.cparam_mode = cparam_mode.lower()
        self.use_pe = use_pe
        self.initialize_h = initialize_h
        self.spatial_smask = spatial_smask
        self.dropout = nn.Dropout(0.1)
        self.use_leff = use_leff
        self.leff_smpl_feats = leff_smpl_feats
        self.leff_fc_in = leff_fc_in
        # =====>> post-process the estimated gait features
        # get extra gaitFeat token from estimated gf + cparams
        gfeat_mpl_input = num_avg_gfeat if not estim_phase else num_avg_gfeat+4
        t_out_dim = self.x_size
        s_out_dim = gfeat_out_channel
        in_dim = self.x_size # for TE
        # add for temporal attention
        self.gfeat_mpl_t = nn.Sequential(
            nn.Linear(gfeat_mpl_input, t_out_dim//2),
            # nn.GELU(),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(t_out_dim//2, t_out_dim),
            nn.Dropout(dropout),
        )
        # concat for spatial attention
        self.gfeat_mpl_s = nn.Sequential(
            nn.Linear(gfeat_mpl_input, s_out_dim//2),
            # nn.GELU(),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(s_out_dim//2, s_out_dim),
            nn.Dropout(dropout),
        )          
        # =====>> construct the temporal encoder <<===== #
        in_dim_s = in_dim + gfeat_out_channel
        # Define `num_token` acc. temporal attn encoder input
        num_token = in_dim//self.unit_size
        # Ensure h_size divisibility
        h_size -= h_size%(num_transformer_head*(num_token+1))
        self.bn_in_s = nn.BatchNorm1d(in_dim_s, momentum=0.1)
        self.bn_in = nn.BatchNorm1d(in_dim, momentum=0.1)
        self.featTencoder = nn.ModuleList([TSAttnBlock(in_dim=in_dim if i==0 else h_size,
                                            encode_dim=h_size, # Q,K,V dim inside encoder
                                            out_dim=in_dim if i==num_layers-1 else h_size, # output size 128*24
                                            num_heads=num_transformer_head,
                                            num_token=num_token,
                                            use_jwff=use_jwff,
                                            ) for i in range(num_layers)])
        self.h_size = h_size
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, cparams, x_smplf=None):
        "Run forward pass for pose feature correction."
        assert x.shape[-1]==self.x_size
        b,n,_ = x.shape
        x_orig = x # it's pose features here
        x_sf_orig = x_smplf
        if self.use_leff and self.leff_smpl_feats:
            assert x_smplf.dim()==5
            B,T,F,S,_ = x_smplf.shape
        # =====>>> estimate gait features
        pred_avg, pred_phase, xc = self.featnet(x, cparams=cparams)

        # =====>>> construct FeatCorrector Input <<<===== #
        # prepare phase
        _, _, p = pred_phase.shape
        norm_phase1 = torch.norm(pred_phase[:,:,:2],dim=-1).unsqueeze(-1).expand(pred_phase.shape[0], pred_phase.shape[1],2)
        norm_phase2 = torch.norm(pred_phase[:,:,2:],dim=-1).unsqueeze(-1).expand(pred_phase.shape[0], pred_phase.shape[1],2)
        phase = pred_phase/torch.cat([norm_phase1,norm_phase2],dim=-1)
        # prepare gait parameters P_G
        _pred_avg = pred_avg.unsqueeze(-2).expand(b, n, pred_avg.shape[-1])
        if self.use_gt_gaitfeat:
            # avoid the post structure to update the P_G estimation
            _pred_avg = _pred_avg.detach()
            phase = phase.detach()
        raw_gfeat = torch.cat([_pred_avg, phase], dim=-1).reshape(b, n, -1)
        # =====>> construct gfeat token
        gfeats_t = self.gfeat_mpl_t(raw_gfeat)
        gfeats_s = self.gfeat_mpl_s(raw_gfeat)
        x_wgf = x + self.dropout(gfeats_t)
        x_wgf_s = torch.cat([x, gfeats_s], dim=-1)                 

        # =====>> process the gait features in TemporalEncoder <<===== #
        # prepare input for temporal attention-based encoder

        y = self.bn_in(x_wgf.transpose(1,2)).transpose(1,2)
        y_s = self.bn_in_s(x_wgf_s.transpose(1,2)).transpose(1,2)

        for i in range(self.num_layers):
            y = self.featTencoder[i](y.reshape(b,N,self.unit_size,-1), \
                xs=y_s.reshape(b,n,self.unit_size,-1))

        y = y[:,:n,:self.x_size] # ignore the prepend token at #-1

        # =====>>> correct the original pose features in a residual way
        y = (y + x_orig).reshape(b*n, -1, self.num_joints)

        if pred_avg is not None: 
            try: pred_avg.retain_grad()
            except: pass
        if pred_phase is not None:
            try: pred_phase.retain_grad()
            except: pass
        return y, pred_avg, pred_phase