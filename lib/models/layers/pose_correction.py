import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from typing import NewType, Optional
from dataclasses import dataclass, asdict, fields
from timm.models.layers import trunc_normal_
from lib.models.layers import LocallyConnected2d
from lib.models.layers.feature_correction import LocomotivePE, TSAttnBlock, PositionalEncoding
from lib.models.layers.gait_feat_encoder import BidirectionalModel, AttentionModel

fun_normalize = lambda x: x/(torch.norm(x,dim=-1,keepdim=True)+1e-12)

Tensor = NewType('Tensor', torch.Tensor)

class PoseCorrector(nn.Module):
    "Args:\n\t`use_gt_gaitfeat`  whether to use ground truth extra features to supervise."
    use_gt_gaitfeat = True
    def __init__(self, 
        input_size,
        fnet_mode='gru',
        num_outputs=3,
        num_layers=1,
        h_size=100,
        estim_phase=True,
        use_residual=True,
        num_joints=24,
        seqlen=90,
        all_concat=False,
        ):
        """
        Construct a PoseCorrector neural network.
        Arguments:
         -- input_size: input dimension, depends on the representation of pose (rot-6D/axis-angle/pose-feature).
         -- num_joints: number of skeleton joints.
         -- TODO model_velocities: force the network to model velocities instead of absolute rotations.
        """
        assert fnet_mode.lower() =='gru'
        super().__init__()

        self.x_size = input_size
        self.featnet = BidirectionalModel(
            seqlen=seqlen,
            num_outputs=num_outputs,
            estime_phase=estim_phase,
            num_joints=num_joints,
            num_layers=num_layers,
            use_pareFeat=False,
        )
        # basic hyperparams
        self.num_joints = num_joints
        self.all_concat = all_concat
        self.estim_phase = estim_phase
        self.h_size = h_size
        self.input_size = 6*num_joints+4 if estim_phase else 6*num_joints
        self.input_size += num_outputs if all_concat else 0
        ###
        self.fc_gf = nn.Linear(num_outputs, h_size) # extend the estimated speed and step
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=h_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.h0 = nn.Parameter(torch.zeros(self.rnn.num_layers*2, 1, h_size).normal_(std=0.001), requires_grad=True)
        self.num_outputs = num_outputs
        self.relu = nn.ReLU(inplace=False)
        self.fc = nn.Linear(self.h_size*2, num_joints*6) # h_size*2 for bidirectional GRU

        self.use_residual = use_residual

    def forward(self, rot6ds, x=None, pred_avg=None, pred_phase=None):
        """
        Run a forward pass for pose corrector.\n
        """
        if x is not None:
            n,t,f = x.shape
            # x = x.permute(1,0,2) # NTF -> TNF
            assert x.shape[-1]==self.x_size
            x_orig = x

            pred_avg, pred_phase, _ = self.featnet(x)
        else:
            n,t,_ = rot6ds.shape
            assert pred_avg is not None and pred_phase is not None
            # h0 = torch.zeros(self.rnn.num_layers*2, n, self.h_size).to(quaters.device)

        if self.estim_phase:
            norm_phase1 = torch.norm(pred_phase[:,:,:2],dim=-1).unsqueeze(-1).expand(pred_phase.shape[0], pred_phase.shape[1],2)
            norm_phase2 = torch.norm(pred_phase[:,:,2:],dim=-1).unsqueeze(-1).expand(pred_phase.shape[0], pred_phase.shape[1],2)
            phase = pred_phase/torch.cat([norm_phase1,norm_phase2],dim=-1)
            if self.use_gt_gaitfeat:
                if self.all_concat:
                    x_cat = torch.cat((rot6ds, phase.detach(), \
                        pred_avg.detach().unsqueeze(1).expand(n,t,pred_avg.shape[-1])), dim=-1)
                else:
                    x_cat = torch.cat((rot6ds, phase.detach()), dim=-1)                   
            else:
                if self.all_concat:
                    x_cat = torch.cat((rot6ds, phase, \
                        pred_avg.unsqueeze(1).expand(n,t, pred_avg.shape[-1])), dim=-1)
                else:
                    x_cat = torch.cat((rot6ds, phase), dim=-1) 
        else:
            x_cat = rot6ds

        if self.use_gt_gaitfeat:
            h0 = self.h0 + self.fc_gf(pred_avg.detach()).unsqueeze(0)
        else: # let the other losses update the pose corrector
            h0 = self.h0 + self.fc_gf(pred_avg).unsqueeze(0)
        # TODO ===> only use self.h0
        h = h0.contiguous()
        y, h = self.rnn(x_cat, h)

        y = self.fc(self.relu(y))

        if self.use_residual:
            y = y + rot6ds
        
        try: # only in training mode
            pred_avg.retain_grad()
            pred_phase.retain_grad()
        except RuntimeError: pass

        return y, pred_avg, pred_phase

class FeatCorrector(nn.Module):
    use_gt_gaitfeat = True
    def __init__(self,
        x_size, # per joint or not (leff_smpl_feats)
        gfeat_out_channel, # dim of extra token
        num_avg_gfeat=3, # number of averaged gait features
        estim_phase=True,
        fnet_mode='gru', # ['gru', 'attn']
        seqlen=90,
        num_layers=2,
        h_size=1024, # TE encode_dim
        num_joints=24,
        temporal_encode='gru',
        use_spatial=False,
        pare_concat=True,
        num_transformer_head=1,
        initialize_h=False,
        use_pe=False,
        cparam_mode='add',
        gf_mode='add',
        smpl_feats_dim=128,
        smpl_patch_size=56,
        smpl_feats_kernel=4,
        cparam_normalize=False,
        spatial_lfc=False,
        spatial_smask=False,
        attn_embed_size=512,
        dropout=0.1,
        use_jwff=False,
        use_leff=False,
        leff_smpl_feats=False,
        leff_fc_in=True,
        leff_h_size_mul=0.5,
        leff_in_dim=256,
        ):
        "Pose Feature correction based on estimated gait features. Always use residual."
        assert fnet_mode.lower() in ['gru', 'attn'] and temporal_encode.lower() in ['gru', 'mae']
        assert cparam_mode.lower() in ['none','add','concat','pe']
        assert gf_mode.lower() in ['add', 'concat','tasc', 'tcsa',]
        # if concatenate, will <=> extra token 
        assert gfeat_out_channel%x_size==0
        super().__init__()
        self.num_avg_gfeat = num_avg_gfeat
        self.num_joints = num_joints
        # ===== GaitFeat module ===== #
        self.fnet_mode = fnet_mode.lower()
        self.tencode_type = temporal_encode.lower()
        self.num_layers = num_layers
        self.unit_size = x_size 
        self.x_size = x_size*num_joints 
        self.pare_concat = pare_concat
        self.with_gfeats = True if num_avg_gfeat>0 or estim_phase else False
        if self.with_gfeats:
            if self.fnet_mode=='gru':
                # most params are fixed
                self.featnet = BidirectionalModel(
                    seqlen=seqlen,
                    input_size=x_size, # per joint
                    num_outputs=num_avg_gfeat,
                    estime_phase=estim_phase,
                    num_joints=num_joints,
                    use_pareFeat=True,
                    pare_concat=pare_concat,
                )
            elif self.fnet_mode=='attn':
                # will always use PareFeat, most params are fixed
                assert num_avg_gfeat==3
                self.featnet = AttentionModel(
                    x_size=x_size,
                    num_joints=num_joints,
                    encode_dim=attn_embed_size,
                )
            else: raise NotImplementedError
        # =====>> Feature (pose) Correction (per frame) <<===== #
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
        if self.with_gfeats:
            if self.fnet_mode == 'gru':
                # get extra gaitFeat token from estimated gf + cparams
                gfeat_mpl_input = num_avg_gfeat if not estim_phase else num_avg_gfeat+4
                if not (self.use_leff and self.leff_smpl_feats) and (self.gf_mode in ['tasc', 'tcsa',]):
                    use_spatial = True
                    if self.gf_mode=='tasc':
                        t_out_dim = self.x_size
                        s_out_dim = gfeat_out_channel
                        in_dim = self.x_size # for TE
                    else: # 'tcsa'
                        t_out_dim = gfeat_out_channel
                        s_out_dim = self.x_size
                        in_dim = self.input_size # for TE
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
                else:
                    if self.use_leff and self.leff_smpl_feats:
                        out_gf_dim = x_size
                    else:
                        if self.gf_mode=='add':
                            out_gf_dim = self.x_size
                            in_dim = self.x_size # for TE
                        elif self.gf_mode=='concat':
                            out_gf_dim = gfeat_out_channel
                            in_dim = self.input_size # for TE
                        else: raise NotImplementedError
                    self.gfeat_mpl = nn.Sequential(
                        nn.Linear(gfeat_mpl_input, out_gf_dim//2),
                        # nn.GELU(),
                        nn.LeakyReLU(0.05, inplace=True),
                        nn.Dropout(dropout),
                        nn.Linear(out_gf_dim//2, out_gf_dim),
                        nn.Dropout(dropout),
                    )
            elif self.fnet_mode == 'attn':
                self.wcyc_pe = LocomotivePE(self.x_size//4, num_in=4, scale=1000.,)
                self.position_encoding = PositionalEncoding(self.x_size//4,)
                self.fc_gf = nn.Linear(h_size, in_dim)
                self.dropout = nn.Dropout(0.1)
                in_dim = self.x_size//4
            else: raise NotImplementedError
        else:
            in_dim = self.x_size
            gfeat_out_channel = 0
        # =====>> construct the temporal encoder <<===== #
        if self.tencode_type=='gru':
            self.featrnn = nn.RNN(in_dim, h_size, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(h_size*2, self.x_size) # X2 for bidirectional
            if self.initialize_h:
                self.avg_mlp = nn.Linear(3, h_size)
        elif self.tencode_type=='mae':
            if self.use_leff and self.leff_smpl_feats:
                in_dim = smpl_feats_dim*((smpl_patch_size//smpl_feats_kernel)**2+1) # per joint, 128
                num_token = num_joints
                self.proj = nn.Sequential(
                    # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(smpl_feats_dim, smpl_feats_dim, kernel_size=smpl_feats_kernel, stride=smpl_feats_kernel,), # get 14x14 output
                    nn.BatchNorm2d(smpl_feats_dim, momentum=0.1,),
                )
                self.pos_embed = nn.Parameter(torch.zeros(1, (smpl_patch_size//smpl_feats_kernel)**2+1, smpl_feats_dim))
                trunc_normal_(self.pos_embed, std=.02)
                # upsample feature maps, spatial restoration
                self.restore = nn.ConvTranspose2d(in_channels=smpl_feats_dim, out_channels=smpl_feats_dim, \
                    kernel_size=smpl_feats_kernel, stride=smpl_feats_kernel,)
            else:
                in_dim_s = in_dim + gfeat_out_channel if self.gf_mode=='tasc' else in_dim - gfeat_out_channel
                if self.cparam_mode=='none': pass
                elif self.cparam_mode=='add' or self.cparam_mode=='concat':
                    # get 'cparam_mlp' for x_t 
                    in_dim = in_dim if 'add' else in_dim + gfeat_out_channel
                    self.cparam_mlp = nn.Sequential(
                        nn.Linear(3, in_dim//2),
                        # nn.GELU(),
                        nn.LeakyReLU(0.05, inplace=True),
                        nn.Dropout(dropout),
                        nn.Linear(in_dim//2, in_dim),
                        nn.Dropout(dropout),
                    )
                    if self.gf_mode in ['tasc', 'tcsa',]: # if has extra input x_s
                        self.cparam_mlp_s = nn.Sequential(
                            nn.Linear(3, in_dim_s//2),
                            # nn.GELU(),
                            nn.LeakyReLU(0.05, inplace=True),
                            nn.Dropout(dropout),
                            nn.Linear(in_dim_s//2, in_dim_s),
                            nn.Dropout(dropout),
                        )
                elif self.cparam_mode=='pe':
                    self.cparam_pe = LocomotivePE(d_feature=in_dim, num_in=3, normalize=cparam_normalize, scale=100., use_fc=False)
                    if self.gf_mode in ['tasc', 'tcsa',]:
                        self.cparam_pe_s = LocomotivePE(d_feature=in_dim_s, num_in=3, normalize=cparam_normalize, scale=100., use_fc=False)

                else: raise NotImplementedError

                if self.initialize_h:
                    self.avg_mlp = nn.Sequential(
                        nn.Linear(3, in_dim//4),
                        nn.LeakyReLU(0.05, inplace=True),
                        nn.Dropout(dropout),
                        nn.Linear(in_dim//4, in_dim),
                    )
                # Define `num_token` acc. temporal attn encoder input
                num_token = in_dim//self.unit_size
                # Ensure h_size divisibility, not only for LFC
                if self.gf_mode=='tasc' and self.with_gfeats:
                    h_size -= h_size%(num_transformer_head*(num_token+1))
                elif self.gf_mode=='tcsa':
                    h_size -= h_size%(num_transformer_head*(num_token-1))
                else:
                    h_size -= h_size%(num_transformer_head*num_token)

                if self.use_pe:
                    self.position_encoding = PositionalEncoding(d_model=in_dim,)
                    if self.gf_mode in ['tasc', 'tcsa',]:
                        self.position_encoding_s = PositionalEncoding(d_model=in_dim_s,)
                if self.use_leff:
                    assert h_size%num_token==0
                if self.gf_mode in ['tasc', 'tcsa',]:
                    self.bn_in_s = nn.BatchNorm1d(in_dim_s, momentum=0.1)
            self.bn_in = nn.BatchNorm1d(in_dim, momentum=0.1)

            self.featTencoder = nn.ModuleList([TSAttnBlock(in_dim=in_dim if i==0 else h_size,
                                                encode_dim=h_size, # Q,K,V dim inside encoder
                                                out_dim=in_dim if i==num_layers-1 else h_size, # output size 128*24
                                                num_heads=num_transformer_head,
                                                num_token=num_token,
                                                use_spatial=use_spatial,
                                                spatial_lfc=spatial_lfc,
                                                split_mode=self.gf_mode if i==0 else 'none',
                                                use_jwff=use_jwff,
                                                use_leff=use_leff,
                                                leff_fc_in=leff_fc_in,
                                                leff_smpl_feats=leff_smpl_feats,
                                                # leff_in_dim=leff_in_dim,
                                                leff_h_size_mul=leff_h_size_mul,
                                                fake_concat=not self.with_gfeats,
                                                ) for i in range(num_layers)])
        else:
            raise NotImplementedError(f"Unknown temporal encoder type {self.tencode_type}!")
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
        # =====>>> estimate gait features, gf_token from 'attn' GF module
        if self.with_gfeats:
            pred_avg, pred_phase, xc = self.featnet(x, cparams=cparams) # xc: with cparam combined

            # =====>>> construct FeatCorrector Input <<<===== #
            if pred_phase is not None:
                _, _, p = pred_phase.shape
                norm_phase1 = torch.norm(pred_phase[:,:,:2],dim=-1).unsqueeze(-1).expand(pred_phase.shape[0], pred_phase.shape[1],2)
                norm_phase2 = torch.norm(pred_phase[:,:,2:],dim=-1).unsqueeze(-1).expand(pred_phase.shape[0], pred_phase.shape[1],2)
                phase = pred_phase/torch.cat([norm_phase1,norm_phase2],dim=-1)
        # estimate phase by default
        if self.with_gfeats:
            if self.fnet_mode=='gru':
                if pred_avg is not None:
                    _pred_avg = pred_avg.unsqueeze(-2).expand(b, n, pred_avg.shape[-1])
                    if self.use_gt_gaitfeat:
                        _pred_avg = _pred_avg.detach()
                    if pred_phase is not None:
                        if self.use_gt_gaitfeat:
                            phase = phase.detach()
                        raw_gfeat = torch.cat([_pred_avg, phase], dim=-1).reshape(b, n, -1)
                    else:
                        raw_gfeat = _pred_avg.reshape(b, n, -1)
                else: 
                    assert pred_phase is not None
                    raw_gfeat = phase.detach().reshape(b, n, -1)
                # =====>> construct gfeat token
                if not (self.use_leff and self.leff_smpl_feats) and (self.gf_mode in ['tasc', 'tcsa',]):
                    gfeats_t = self.gfeat_mpl_t(raw_gfeat)
                    gfeats_s = self.gfeat_mpl_s(raw_gfeat)
                    if self.gf_mode=='tasc':
                        x_wgf = x + self.dropout(gfeats_t)
                        x_wgf_s = torch.cat([x, gfeats_s], dim=-1)
                    else:
                        x_wgf = torch.cat([x, gfeats_t], dim=-1)
                        x_wgf_s = x + self.dropout(gfeats_s)                    
                else:
                    gfeats = self.gfeat_mpl(raw_gfeat)
                    if self.use_leff and self.leff_smpl_feats:
                        x_smplf = self.proj(x_smplf.reshape(B*T,F,S,S)).flatten(2).transpose(1,2)
                        x_wgf = torch.cat([x_smplf, gfeats.reshape(b*n,1,self.unit_size)], dim=1)
                        # add ViT-like positional encoding
                        x_wgf = x_wgf + self.pos_embed
                    else:
                        if self.gf_mode=='add':
                            x_wgf = x + self.dropout(gfeats)
                        elif self.gf_mode=='concat':
                            x_wgf = torch.cat([x, gfeats], dim=-1)
                        x_wgf_s = None
        else:
            x_wgf = x
            x_wgf_s = x
        # =====>> process the gait features in TemporalEncoder <<===== #
        if self.initialize_h and pred_avg is not None:
            avg_token = self.avg_mlp(pred_avg)
        if self.tencode_type=='gru':
            # since concatenate, did not transpose dim
            if self.initialize_h and pred_avg is not None:
                h0 = torch.zeros(self.featrnn.num_layers*2, n, self.h_size).to(x.device)
                h = (h0 + avg_token).contiguous()
                y, _ = self.featrnn(x_wgf, h.detach())
            else:
                y, _ = self.featrnn(x_wgf,)
            y = self.fc(self.relu(y))
        elif self.tencode_type=='mae':
            if self.fnet_mode=='gru':
                # =====>> prepare input for temporal attention-based encoder
                if self.use_leff and self.leff_smpl_feats:
                    y = x_wgf.transpose(1,2).reshape(B,T,F,-1)
                    y_s = None
                else:
                    if self.cparam_mode=='none':
                        x_wcp = x_wgf
                        x_wcp_s = x_wgf_s
                    elif self.cparam_mode in ['add', 'concat',]:
                        cparam_token = self.cparam_mlp(cparams.reshape(b,n,3))
                        if x_wgf_s is not None:
                            cparam_token_s = self.cparam_mlp_s(cparams.reshape(b,n,3))
                        if self.cparam_mode=='add':
                            x_wcp = x_wgf + self.dropout(cparam_token)
                            if x_wgf_s is not None:
                                x_wcp_s = x_wgf_s + self.dropout(cparam_token_s)
                            else: x_wcp_s = None
                        elif self.cparam_mode=='concat':
                            x_wcp = torch.cat([x_wgf, cparam_token], dim=-1)
                            if x_wgf_s is not None:
                                x_wcp_s = torch.cat([x_wgf, cparam_token_s], dim=-1)
                            else: x_wcp_s = None
                    elif self.cparam_mode=='pe':
                        x_wcp = self.cparam_pe(x_wgf.reshape(b,n,-1), feats=cparams.reshape(b,n,3)).reshape(b,n,-1)
                        if x_wgf_s is not None:
                            x_wcp_s = self.cparam_pe_s(x_wgf_s.reshape(b,n,-1), feats=cparams.reshape(b,n,3)).reshape(b,n,-1)
                        else: x_wcp_s = None
                    else:
                        raise NotImplementedError
                    
                    if self.initialize_h:
                        y = torch.cat([x_wcp, avg_token.unsqueeze(1)], dim=1)
                        N = n+1
                    else:
                        y = x_wcp
                        N = n
                    
                    if self.use_pe:
                        y = self.position_encoding(y)
                        if x_wcp_s is not None:
                            y_s = self.position_encoding_s(x_wcp_s)
                    else:
                        y_s = x_wcp_s
                # if not self.leff_smpl_feats:
                y = self.bn_in(y.transpose(1,2)).transpose(1,2)
                if y_s is not None:
                    y_s = self.bn_in_s(y_s.transpose(1,2)).transpose(1,2)

            elif self.fnet_mode=='attn':
                raise NotImplementedError

            for i in range(self.num_layers):
                if self.use_leff and self.leff_smpl_feats:
                    y = self.featTencoder[i](y,).reshape(B,T,self.unit_size,-1,)
                else:
                    y = self.featTencoder[i](y.reshape(b,N,self.unit_size,-1), \
                        xs=y_s.reshape(b,n,self.unit_size,-1) if (y_s is not None) else None)
            if self.use_leff and self.leff_smpl_feats:
                y = y[:,:,:,:-1].transpose(2,3).reshape(B*T,F,-1) # b,n,128,14*14(196)
                res = int(math.sqrt(y.shape[-1]))
                y = self.restore(y.reshape(B*T,F,res,res))
            else:
                y = y[:,:n,:self.x_size] # ignore the prepend token at #-1 for fnet_mode=='attn'

        # =====>>> correct the original pose features in a residual way
        if self.use_leff and self.leff_smpl_feats:
            y = (y.reshape(B,T,F,S,S) + x_sf_orig)
        else:
            y = (y + x_orig).reshape(b*n, -1, self.num_joints)

        if self.with_gfeats:
            if pred_avg is not None: 
                try: pred_avg.retain_grad()
                except: pass
            if pred_phase is not None:
                try: pred_phase.retain_grad()
                except: pass
            return y, pred_avg, pred_phase
        else:
            return y, None, None