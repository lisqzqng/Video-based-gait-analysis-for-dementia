# Author: Diwei Wang 

import math
from turtle import forward
import torch
import numpy as np
import os.path as osp
import logging, sys
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet

from lib.core.config import GRNET_DATA_DIR, SMPL_DATA_DIR
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat, quat2mat
from lib.models.smpl import SMPL_MEAN_PARAMS, SMPLHead, H36M_TO_J14, SMPL, SMPL_MODEL_DIR
from lib.models.hrnet import hrnet_w32
from lib.utils.utils import load_pretrained_model, load_ckpt_w_prefix
from lib.models.pare import PareHead, VPRegressor
from lib.models.layers import FeatCorrector


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class GRNet(nn.Module):
    is_demo = False # to control summary writer
    def __init__(
        self,
        num_joints=24,
        num_input_features=480, # channel num of hrnet output
        num_features_pare=128,
        num_features_smpl=64,
        backbone='hrnet_w32',
        focal_length=5000.,
        img_res=224,
        pretrained_pare=osp.join(GRNET_DATA_DIR, 'pare_w_3dpw_checkpoint.ckpt'),
        writer=None,
        seqlen=50,
        pretrained_hrnet=osp.join(GRNET_DATA_DIR,'hrnet_w32.pth.tar'),
        use_gait_feat=False,
        featcorr=None,
        use_pose_encoder=False,
        use_shpcam_encoder=False,
    ):
        super(GRNet, self).__init__()
        self.focal_length = focal_length
        self.use_pose_encoder = use_pose_encoder
        self.use_shpcam_encoder = use_shpcam_encoder
        self.num_features_pare = num_features_pare
        self.num_joints = num_joints

        self.backbone = eval(backbone)(
            pretrained=True,
            pretrained_ckpt=pretrained_hrnet,
            downsample=False,
            use_conv=True,
        )

        self.head = PareHead(
            num_joints=num_joints,
            seqlen=seqlen,
            num_input_features=num_input_features,
            num_features_pare=num_features_pare, 
            num_features_smpl=num_features_smpl,
        )            
        # ======== Module for locomotive features ======== #
        self.use_gait_feat = use_gait_feat
        if use_gait_feat:
            # make corrections on pose features
            self.pfeat_corrector = FeatCorrector(
                x_size=num_features_pare,
                num_avg_gfeat=featcorr.AVG_DIM,
                seqlen=seqlen,
                num_layers=featcorr.NUM_LAYERS,
                estim_phase=featcorr.ESTIM_PHASE,
                num_joints=num_joints,
                h_size=featcorr.H_SIZE, # for both GRU & Transformer
                num_transformer_head=featcorr.NUM_HEADS,
                use_jwff=featcorr.USE_JWFF,
            )
        else: self.pfeat_corrector = None
        self.regressor = VPRegressor(
            focal_length=focal_length,
            img_res=img_res,
        )
        # if vpare pretrained model exists, load explicitly in demo_dw.py
        assert pretrained_pare and osp.isfile(pretrained_pare), logger.error(f"No pretrained pare weights at {pretrained_pare}.")
        self.pretrained_pare = pretrained_pare
        self.load_pare_dict() # always preload part attention
        if not self.is_demo: self.writer = writer
        self.iteration = 0

    def load_pare_dict(self, pretrained_pare=None):
        "Load pretrained weights from PARE model"
        if pretrained_pare is not None:
            self.pretrained_pare = pretrained_pare
        logger.info(f'Loading pretrained part attention from {self.pretrained_pare}')
        try:
            ckpt = torch.load(self.pretrained_pare)['state_dict']
            self.init_params = {
                'init_pose': ckpt['model.head.init_pose'],
                'init_shape': ckpt['model.head.init_shape'],
            }
        except KeyError:
            logger.error(f"Checkpoint at {self.pretrained_pare} does not match VPARE implementation.")
            sys.exit()
        else:
            load_ckpt_w_prefix(self.head, ckpt, prefix='head.')
            logger.info(f'Loaded pretrained part attention from \"{self.pretrained_pare}\"')

    def load_featnet_dict(self, pretrained_featnet, prefix):
        "Load pretrained weights from FeatNet model"
        logger.info(f'Loading pretrained featnet from {pretrained_featnet}')
        try:
            ckpt = torch.load(pretrained_featnet) # already 'state_dict'
        except:
            logger.error(f"Checkpoint at {pretrained_featnet} does not match the implementation.")
            sys.exit()
        else:
            if prefix=='pfeat_corrector':
                if self.pfeat_corrector.with_gfeats:
                    load_ckpt_w_prefix(self.pfeat_corrector.featnet, ckpt, prefix=f'{prefix}.featnet')
                else:
                    logger.info(f'No gait parameter is used !!')
                    return
            else: raise ValueError('prefix should be in [pfeat_corrector,] !!')
            logger.info(f'Loaded pretrained featnet weights from \"{pretrained_featnet}\"')

    def forward(self, features, bbox=None, cimg=None, J_regressor=None, ):
        # input size (batch, seqlen, 3, 224, 224)
        device = features.device
        if self.use_gait_feat:
            assert (bbox is not None) and (cimg is not None) 
            if len(bbox.shape)==2: bbox = bbox.unsqueeze(0)
            if len(cimg.shape)==2: cimg = cimg.unsqueeze(0)
        if features.dim() == 5: 
            batch_size, seqlen, nc, h, w = features.shape
            features = features.reshape(-1,nc,h,w)
        elif features.dim() == 4:
            batch_size = 1
            seqlen, nc, h, w = features.shape
        else:
            raise ValueError(f"Wrong feature dimension: {features.dim()}.")

        with torch.no_grad():
            features = self.backbone(features)

        point_local_feat, cam_shape_feats, output = self.head.feature_extractor(features=features)
        pose_dim = point_local_feat.shape # (b,seqlen,128,24)
        cs_dim = cam_shape_feats.shape

        patt_output = self.head(point_local_feat, cam_shape_feats, output)
        # quaters = patt_output['pred_quater']
        if self.use_gait_feat:
            cam_params = patt_output['pred_cam']
            bs = bbox[:,:,2]/224.0
            t_bb = bbox[:,:,:2]-cimg
            scale = (bs.reshape(-1,1)*cam_params[:,0:1])
            cparams = torch.cat([scale, t_bb.reshape(-1,2)/scale/112.0 + cam_params[:,1:]], dim=-1)
            # ===== Full Double Correction (re-estimate gait features) ===== #
            assert self.pfeat_corrector is not None
            new_point_local_feat, pred_avg, pred_phase = \
                self.pfeat_corrector(point_local_feat.reshape(batch_size, seqlen,-1), cparams=cparams.reshape(batch_size, seqlen, 3))
            del point_local_feat
            del patt_output
            # regenerate patt_output, including new pred_pose 
            patt_output = self.head(new_point_local_feat, cam_shape_feats, output)
            patt_output['pred_avg'] = pred_avg
            patt_output['pred_phase'] = pred_phase

        output = self.regressor(patt_output, batch_size=batch_size, J_regressor=J_regressor)
        if self.use_gait_feat:
            output[-1]['pred_cparam'] = cparams
        
        return output