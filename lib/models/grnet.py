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

from lib.core.config import VPARE_DATA_DIR, VIBE_DATA_DIR
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat, quat2mat
from lib.models.smpl import SMPL_MEAN_PARAMS, SMPLHead, H36M_TO_J14, SMPL, SMPL_MODEL_DIR
from lib.models.hrnet import hrnet_w32
from lib.utils.utils import load_pretrained_model, load_ckpt_w_prefix
from lib.models.pare import PareHead, VPRegressor
from lib.models.layers import PoseCorrector, FeatCorrector


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class GRNet(nn.Module):
    is_demo = False # to control summary writer
    def __init__(
        self,
        n_layers=2,
        add_linear=False,
        bidirectional=False,
        use_residual=True,
        num_joints=24,
        num_input_features=480, # channel num of hrnet output
        num_features_pare=128,
        num_features_smpl=64,
        backbone='hrnet_w32',
        focal_length=5000.,
        img_res=224,
        pretrained_pare=osp.join(VPARE_DATA_DIR, 'pare_w_3dpw_checkpoint.ckpt'),
        writer=None,
        use_pose_encoder=False,
        use_shpcam_encoder=False,
        #use_rot6d=False,
        seqlen=48,
        pretrained_hrnet=osp.join(VPARE_DATA_DIR,'hrnet_w32.pth.tar'),
        use_gait_feat=False,
        gfeat_num_outputs=3,
        pose_correction=None,
        estim_phase=False,
    ):
        super(GRNet, self).__init__()
        self.focal_length = focal_length
        self.use_pose_encoder = use_pose_encoder
        self.use_shpcam_encoder = use_shpcam_encoder
        #self.use_rot6d = use_rot6d
        self.num_features_pare = num_features_pare
        self.num_joints = num_joints
        self.gfeat_num_outputs = gfeat_num_outputs
        self.smpl_psize = img_res//4

        # if self.is_demo:
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

        if use_pose_encoder:
            # 2 separate encoders for point_local_feat & cam_shape_feat 
            self.pos_encoder = TemporalEncoder(
                n_layers=n_layers,
                input_size=num_features_pare*num_joints,
                hidden_size=int(num_features_pare*num_joints/2) if bidirectional else int(num_features_pare*num_joints),
                bidirectional=bidirectional,
                add_linear=add_linear,
                use_residual=use_residual,
            )
        if use_shpcam_encoder:
            self.cs_encoder = TemporalEncoder(
                n_layers=n_layers,
                input_size=num_features_smpl*num_joints,
                hidden_size=int(num_features_smpl*num_joints/2) if bidirectional else int(num_features_smpl*num_joints),
                bidirectional=bidirectional,
                add_linear=add_linear,
                use_residual=use_residual,
            )
        # ======== Module for locomotive features ======== #
        self.use_gait_feat = use_gait_feat
        if use_gait_feat:
            self.use_smpl_feats = False
            if pose_correction.WithFeat:
                # make corrections on pose features
                self.pfeat_corrector = FeatCorrector(
                    x_size=num_features_pare,
                    gfeat_out_channel=pose_correction.gfeat_out_channel, # dim of extra token
                    num_avg_gfeat=gfeat_num_outputs,
                    fnet_mode=pose_correction.FNET,
                    seqlen=seqlen,
                    num_layers=pose_correction.NUM_LAYERS,
                    estim_phase=estim_phase,
                    num_joints=num_joints,
                    h_size=pose_correction.FEAT_H_SIZE, # for both GRU & Transformer
                    pare_concat=pose_correction.PARE_CONCAT,
                    temporal_encode=pose_correction.TE,
                    use_spatial=pose_correction.USE_SPATIAL,
                    num_transformer_head=pose_correction.NUM_HEADS,
                    use_pe=pose_correction.USE_PE,
                    cparam_mode=pose_correction.CPARAM_MODE,
                    gf_mode=pose_correction.GF_MODE,
                    use_jwff=pose_correction.USE_JWFF,
                    use_leff=pose_correction.USE_LEFF,
                    leff_smpl_feats=pose_correction.LEFF_SMPL_FEATS,
                    leff_fc_in=pose_correction.LEFF_FC_IN,
                    leff_h_size_mul=pose_correction.LEFF_H_SIZE_MUL,
                    leff_in_dim=pose_correction.LEFF_IN_DIM,
                    smpl_feats_dim=self.num_features_pare,
                    cparam_normalize=pose_correction.CPARAM_NORMALIZE,
                    spatial_lfc=pose_correction.SPATIAL_LFC,
                    spatial_smask=pose_correction.SPATIAL_SMASK,
                )
                if pose_correction.USE_LEFF and pose_correction.LEFF_SMPL_FEATS:
                    self.use_smpl_feats = True
            else: self.pfeat_corrector = None
            if pose_correction.WithPose:
                # make corrections on poses TODO 6D v.s. quaternion
                self.pose_corrector = PoseCorrector(
                    input_size=3*num_joints+3,
                    fnet_mode=pose_correction.FNET,
                    num_outputs=gfeat_num_outputs,
                    num_layers=pose_correction.NUM_LAYERS,
                    use_residual=pose_correction.POSE_RESIDUAL,
                    h_size=pose_correction.POSE_H_SIZE,
                    num_joints=num_joints,
                    seqlen=seqlen,
                    estim_phase=estim_phase,
                    all_concat=pose_correction.ALL_CONCAT,
                )
            else: self.pose_corrector = None
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
            elif prefix=='pose_corrector':
                load_ckpt_w_prefix(self.pose_corrector.featnet, ckpt, prefix=f'{prefix}.featnet')
            else: raise ValueError('prefix should be in [pfeat_corrector, pose_corrector] !!')
            logger.info(f'Loaded pretrained featnet weights from \"{pretrained_featnet}\"')

    def feature_extractor(self, images,):
        "get pose features for standalone GaitFeatNet training."
        return self(images, with_pareFeat=True)[-1]['pareFeat']

    def forward(self, features, bbox=None, cimg=None, J_regressor=None, with_pareFeat=False):
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
            
        # ============>> GRU-based temporal encoder
        # =====================>> Naive pose corrector
        #                         not incorporated into pose TE
        if self.use_pose_encoder:
            point_local_feat = point_local_feat.reshape(batch_size, seqlen, -1)
            point_local_feat = self.pos_encoder(point_local_feat).reshape(pose_dim)

        if self.use_shpcam_encoder:
            cam_shape_feats = cam_shape_feats.reshape(batch_size, seqlen, -1)
            cam_shape_feats = self.cs_encoder(cam_shape_feats).reshape(cs_dim)

        visualize=False
        if visualize:
            part_attention = output['pred_segm_mask'][:,1:]
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
            import torch.nn.functional as F
            import colorsys, cv2
            batch_size, num_joints, height, width = part_attention.shape
            norm_heatmap = F.softmax(part_attention.reshape(batch_size,num_joints, -1),dim=-1).reshape(batch_size, num_joints, height, width).cpu()
            colorlist = np.linspace(0.1,1,num_joints,endpoint=False)
            for seq in range(int(seqlen/2), seqlen):
                img = images[0,seq].permute(1,2,0).cpu().numpy()
                for k in range(num_joints):
                    mp = norm_heatmap[seq,k]*height*width
                    mp = torch.stack([torch.ones((height, width))*colorlist[k], \
                        torch.ones((height, width)), mp], dim=2).numpy()
                    mp = cv2.cvtColor(mp.copy(), cv2.COLOR_HSV2RGB)
                    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
                    plt.subplot(outer[0])
                    plt.imshow(img)
                    plt.subplot(outer[1])
                    plt.title(f'joint: {k}')
                    plt.imshow((mp*255).astype(np.int64))
                    plt.show()

        patt_output = self.head(point_local_feat, cam_shape_feats, output)
        # quaters = patt_output['pred_quater']
        if self.use_gait_feat:
            cam_params = patt_output['pred_cam']
            bs = bbox[:,:,2]/224.0
            t_bb = bbox[:,:,:2]-cimg
            scale = (bs.reshape(-1,1)*cam_params[:,0:1])
            cparams = torch.cat([scale, t_bb.reshape(-1,2)/scale/112.0 + cam_params[:,1:]], dim=-1)
            # ===== Full Double Correction (re-estimate gait features) ===== #
            if self.pfeat_corrector is not None:
                if self.use_smpl_feats:
                    # get smpl_feats from backbone-extracted features
                    smpl_feats = output['smpl_feats'].reshape(batch_size, seqlen, self.num_features_pare, self.smpl_psize, self.smpl_psize)
                    new_smpl_feats, pred_avg, pred_phase = self.pfeat_corrector(point_local_feat.reshape(batch_size, seqlen,-1), \
                        cparams=cparams.reshape(batch_size, seqlen, 3), x_smplf=smpl_feats,)
                    del smpl_feats
                    output['smpl_feats'] = new_smpl_feats.reshape(batch_size*seqlen, \
                        self.num_features_pare, self.smpl_psize, self.smpl_psize)
                    new_point_local_feat, cam_shape_feats, output = self.head.feature_extractor(output=output)
                    patt_output = self.head(new_point_local_feat, cam_shape_feats, output)
                else:
                    new_point_local_feat, pred_avg, pred_phase = \
                        self.pfeat_corrector(point_local_feat.reshape(batch_size, seqlen,-1), cparams=cparams.reshape(batch_size, seqlen, 3))
                    del point_local_feat
                    del patt_output
                    # regenerate patt_output, including new pred_pose 
                    patt_output = self.head(new_point_local_feat, cam_shape_feats, output)
                if self.pose_corrector is not None:
                    # use same gait features in pfeat_corrector, if both corrections are called
                    root_axisang = rotation_matrix_to_angle_axis(patt_output['pred_pose'][:,0])
                    rot6ds = (patt_output['pred_pose'][:,:,:,:2].reshape(batch_size,seqlen,-1)) # get 6D repres' from rotmat
                    rot6ds, _, _ = \
                        self.pose_corrector(rot6ds=rot6ds, pred_avg=pred_avg, pred_phase=pred_phase)
                    patt_output['pred_pose'] = rot6d_to_rotmat(rot6ds).reshape(-1,self.num_joints,3,3)
            elif self.pose_corrector is not None:
                root_axisang = rotation_matrix_to_angle_axis(patt_output['pred_pose'][:,0])
                rot6ds = (patt_output['pred_pose'][:,:,:,:2].reshape(batch_size,seqlen,-1))
                body_joints = self.regressor.get_body_joints(patt_output)             
                njoints = torch.cat([root_axisang, body_joints[:,1:,:].reshape(-1,(self.num_joints-1)*3), cparams], dim=-1)
                rot6ds, pred_avg, pred_phase = \
                    self.pose_corrector(rot6ds=rot6ds,x=njoints.reshape(batch_size,seqlen,-1))
                patt_output['pred_pose'] = rot6d_to_rotmat(rot6ds).reshape(-1,self.num_joints,3,3)
                
            else: raise AssertionError("No pose correction mode is specified!!")
            patt_output['pred_avg'] = pred_avg
            patt_output['pred_phase'] = pred_phase

        output = self.regressor(patt_output, batch_size=batch_size, J_regressor=J_regressor)#, use_rot6d=self.use_rot6d)
        if self.use_gait_feat:
            output[-1]['pred_cparam'] = cparams
        if with_pareFeat:
            output[-1]['pareFeat'] = point_local_feat
        
        return output