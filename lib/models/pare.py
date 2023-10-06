# This script is hacked from https://github.com/mkocabas/PARE
# Please adhere to their li_cense to use this script

import math
from turtle import forward
import torch
import numpy as np
import os.path as osp
import logging, sys
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet

from lib.core.config import SMPL_DATA_DIR
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from lib.models.smpl import SMPL_MEAN_PARAMS, SMPLHead, H36M_TO_J14, SMPL, SMPL_MODEL_DIR
from lib.models.layers import LocallyConnected2d, KeypointAttention
from lib.utils.geometry import projection
from lib.utils.geometry import quat2mat, rotation_matrix_to_quaternion

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class VPRegressor(nn.Module):
    def __init__(
            self,
            focal_length=5000.,
            img_res=224,
    ):
        super(VPRegressor, self).__init__()
        # smpl mean params are loaded by SMPLHead, use SMPL_MODEL_DIR by default
        self.smpl = SMPLHead(
            focal_length=focal_length,
            img_res=img_res,
            smpl_model_dir=SMPL_MODEL_DIR,
        )

    def get_body_joints(self, patt_output, batch_size=1, J_regressor=None):
        shape = patt_output['pred_pose'].shape
        global_orient = torch.zeros_like(patt_output['pred_pose'][:, 0].unsqueeze(1))
        global_orient[:,:,] = torch.eye(3)
        smpl_output = self.smpl(
            betas=patt_output['pred_shape'],
            body_pose=patt_output['pred_pose'][:,1:],#rotmat
            global_orient=global_orient,
            pose2rot=False,
        )
        njoints = smpl_output.joints

        return njoints.reshape(shape[0], shape[1], -1) # BNF

    def forward(self, patt_output, batch_size=1, J_regressor=None,): #use_rot6d=False):
        """ Use SMPLHead instead of SMPL
        """
        smpl_output = self.smpl(
            rotmat=patt_output['pred_pose'],#rotmat
            shape=patt_output['pred_shape'],
            cam=patt_output['pred_cam'],
            normalize_joints2d=True,
        )
        smpl_output.update(patt_output)
        # pred_pose is already rotmat
        pred_rotmat = smpl_output['pred_pose']
        # if use_rot6d:
            # pose = patt_output['pred_rot6d'].reshape(-1,144)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        # pose = rotation_matrix_to_quaternion(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 96) #quaternion
        seqlen = int(pose.shape[0]/batch_size)

        if J_regressor is not None:
            pred_vertices = smpl_output['smpl_vertices'].reshape(batch_size*seqlen, -1, 3)
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            if J_regressor.shape[0]<24:
                pred_joints = pred_joints[:, H36M_TO_J14, :]
            smpl_output['smpl_joints3d'] = pred_joints

        output = [{
            'theta'  : torch.cat([smpl_output['pred_cam'], pose, smpl_output['pred_shape']], dim=1).reshape(batch_size, seqlen, -1),
            'verts'  : smpl_output['smpl_vertices'].reshape(batch_size, seqlen, -1, 3),
            'kp_2d'  : smpl_output['smpl_joints2d'].reshape(batch_size, seqlen, -1, 2),
            'kp_3d'  : smpl_output['smpl_joints3d'].reshape(batch_size, seqlen, -1, 3),
            'rotmat' : pred_rotmat.reshape(batch_size, seqlen, -1, 3, 3),
        }]
        try: output[-1].update({
            'pred_avg': patt_output['pred_avg'],
            'pred_phase': patt_output['pred_phase'],
        })
        except: pass

        return output

class SMPLRegressor(nn.Module):
    "get 2D/3D joints and vertices from SMPL params"
    def __init__(
            self,
            focal_length=5000.,
            img_res=224,
    ):
        super(SMPLRegressor, self).__init__()
        # smpl mean params are loaded by SMPLHead, use SMPL_MODEL_DIR by default
        self.smpl = SMPLHead(
            focal_length=focal_length,
            img_res=img_res,
            smpl_model_dir=SMPL_MODEL_DIR,
        )

    def forward(self, patt_output, batch_size=1, J_regressor=None,): #use_rot6d=False):
        """ Use SMPLHead instead of SMPL
        """
        smpl_output = self.smpl(
            rotmat=patt_output['pred_rotmat'],#rotmat
            shape=patt_output['pred_shape'],
            cam=patt_output['pred_cam'],
            normalize_joints2d=True,
        )
        smpl_output.update(patt_output)
        # pred_pose is already rotmat
        pred_rotmat = smpl_output['pred_rotmat']
        # if use_rot6d:
            # pose = patt_output['pred_rot6d'].reshape(-1,144)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        # pose = rotation_matrix_to_quaternion(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 96) #quaternion
        seqlen = int(pose.shape[0]/batch_size)

        if J_regressor is not None:
            pred_vertices = smpl_output['smpl_vertices'].reshape(batch_size*seqlen, -1, 3)
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            if J_regressor.shape[0]<24:
                pred_joints = pred_joints[:, H36M_TO_J14, :]
            smpl_output['smpl_joints3d'] = pred_joints
        
        # standardize output
        output = {
            'kp_2d'  : smpl_output['smpl_joints2d'].reshape(batch_size, seqlen, -1, 2),
            'kp_3d'  : smpl_output['smpl_joints3d'].reshape(batch_size, seqlen, -1, 3),
            'rotmat' : pred_rotmat.reshape(batch_size, seqlen, -1, 3, 3),
            'verts'  : smpl_output['smpl_vertices'].reshape(batch_size, seqlen, -1, 3),
            }

        return output

class PareHead(nn.Module):
    def __init__(self,
            num_joints,
            num_input_features,
            seqlen=48,
            softmax_temp=1.0,
            num_iterations=3,
            use_resnet_conv_hrnet=False,
            num_deconv_layers=2,
            num_deconv_kernels=(4, 4),
            num_camera_params=3,
            num_features_pare=128,
            num_features_smpl=64,
            final_conv_kernel=1,
            pose_mlp_num_layers=1,
            shape_mlp_num_layers=1,
            pretrained_ckpt=None,
            iterative_regression=False,
    ):
        super(PareHead, self).__init__()
        self.use_heatmaps = 'part_segm'
        self.deconv_with_bias = False
        self.pose_mlp_num_layers = pose_mlp_num_layers
        self.shape_mlp_num_layers = shape_mlp_num_layers
        self.pose_mlp_hidden_size = 256
        self.shape_mlp_hidden_size = 256
        self.num_input_features = num_input_features
        self.num_joints = num_joints
        self.seqlen = seqlen
        self.num_iterations = 1
        assert isinstance(num_input_features, int), f"Unknown type of num_input_features: {type(num_input_features)}."
        num_deconv_filters = []
        for ind in range(num_deconv_layers):
            num_deconv_filters.append(num_features_pare)

        assert num_iterations > 0, '\"num_iterations\" should be greater than 0.'

        self.keypoint_deconv_layers = self._make_conv_layer(
            num_deconv_layers,
            num_deconv_filters,
            (3,)*num_deconv_layers,
        )
        self.num_input_features = num_input_features
        self.smpl_deconv_layers = self._make_conv_layer(
            num_deconv_layers,
            num_deconv_filters,
            (3,)*num_deconv_layers,
        )
        self.iterative_regression = iterative_regression
        self.pose_mlp_inp_dim = num_deconv_filters[-1]
        smpl_final_dim = num_features_smpl
        self.shape_mlp_inp_dim = num_joints * smpl_final_dim
        # --------------------->> Not self.use_soft_attention
        self.keypoint_final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=num_joints+1, # Since `use_heatmaps == 'part_segm'`
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0,
        )
        self.smpl_final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=smpl_final_dim,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0,
        )

        # temperature for softargmax function
        self.register_buffer('temperature', torch.tensor(softmax_temp))

        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        # --------------------->> Not self.use_hmr_regression
        # here we use 2 different MLPs to estimate shape and camera
        # They take a channelwise downsampled version of smpl features
        self.shape_mlp = self._get_shape_mlp(output_size=10)
        self.cam_mlp = self._get_shape_mlp(output_size=num_camera_params)

        # for pose each joint has a separate MLP
        # weights for these MLPs are not shared
        # hence we use Locally Connected layers
        # TODO support kernel_size > 1 to access context of other joints
        self.pose_mlp = self._get_pose_mlp(num_joints=num_joints, output_size=6)

        # --------------------->> self.use_keypoint_attention
        # output attended_features
        self.keypoint_attention = KeypointAttention(
            use_conv=False,#use_postconv_keypoint_attention,
            in_channels=(self.pose_mlp_inp_dim, smpl_final_dim),
            out_channels=(self.pose_mlp_inp_dim, smpl_final_dim),
            act='softmax',#keypoint_attention_act,
            use_scale=False,#use_scale_keypoint_attention,
        )

    def feature_extractor(self, features=None, output=None, basicfeat=False,):
        # x is the output of backbone hrnet
        assert not (features is None and output is None)
        if output is None:
            output = {}
            ############## 2D PART BRANCH FEATURES ##############
            part_feats = self._get_2d_branch_feats(features) # part_feats is the output of ReLU, >0
            ############## GET PART ATTENTION MAP ##############
            part_attention = self._get_part_attention_map(part_feats, output) # part_attention can take negative values
            ############## 3D SMPL BRANCH FEATURES ##############
            smpl_feats = self._get_3d_smpl_feats(features, part_feats) # smpl_feats is the output of ReLU, >0
            output['smpl_feats'] = smpl_feats
            output['part_attn'] = part_attention
            output['part_feats'] = part_feats
            if basicfeat:
                return output
            else:
                point_local_feat, cam_shape_feats = self._get_local_feats(smpl_feats, part_attention, output)
        else:
            # already got the part/smpl_feats and part_attention_mask, apply the key_attention
            assert torch.is_tensor(output['smpl_feats']) and torch.is_tensor(output['part_attn'])
            assert not basicfeat
            point_local_feat, cam_shape_feats = self._get_local_feats(output['smpl_feats'], output['part_attn'], output)
            return point_local_feat, cam_shape_feats
    
    def forward(self, point_local_feat, cam_shape_feats, output, inits=None, gt_segm=None):
        ############## GET FINAL PREDICTIONS ##############
        batch_size = point_local_feat.shape[0]

        if inits is None:
            init_pose = self.init_pose.expand(batch_size, -1)  # N, Jx6
            init_shape = self.init_shape.expand(batch_size, -1)
            init_cam = self.init_cam.expand(batch_size, -1)
            iter_now = False # use direct MLP to regress the very initial params
        else:
            init_pose = inits['pred_rot6d']
            init_shape = inits['pred_shape']
            init_cam = inits['pred_cam']
            iter_now = True
        
        pred_pose, pred_shape, pred_cam = self._pare_get_final_preds(
            point_local_feat, cam_shape_feats, init_pose, init_shape, init_cam, iter_now=iter_now,
        )
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).reshape(batch_size, 24, 3, 3)
        # pred_quater = rotation_matrix_to_quaternion(pred_rotmat.reshape(-1,3,3)).reshape(-1,24,4)
        # pred_pose_rot6d = pred_pose.reshape(-1,24,3,2)/(torch.norm(pred_pose.reshape(-1,24,3,2),dim=-2,keepdim=True))

        output.update({
            'pred_rotmat': pred_rotmat,
            # 'pred_quater': pred_quater.reshape(-1,self.num_joints,4),
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_rot6d': pred_pose, # 6D rotation without normalization
        })

        return output

    def _get_2d_branch_feats(self, features):
        part_feats = self.keypoint_deconv_layers(features) # [Conv2d, BN, ReLU]x2
        # if self.use_branch_nonlocal:
        #     part_feats = self.branch_2d_nonlocal(part_feats)
        return part_feats

    def _get_part_attention_map(self, part_feats, output):
        heatmaps = self.keypoint_final_layer(part_feats) # 1 Conv2d layer, weight & bias can take pos and neg values

        # self.use_heatmaps == 'part_segm'
        output['pred_segm_mask'] = heatmaps
        heatmaps = heatmaps[:,1:,:,:]
        return heatmaps

    def _get_3d_smpl_feats(self, features, part_feats):
        # if self.use_keypoint_features_for_smpl_regression:
        #     smpl_feats = part_feats
        # else:
        smpl_feats = self.smpl_deconv_layers(features) # [Conv2d, BN, ReLU]x2
        # if self.use_branch_nonlocal:
        #     smpl_feats = self.branch_3d_nonlocal(smpl_feats)
        return smpl_feats
    
    def _get_local_feats(self, smpl_feats, part_attention, output=None):
        cam_shape_feats = self.smpl_final_layer(smpl_feats) # 1 Conv2d layer
        # if self.use_keypoint_attention:
        point_local_feat = self.keypoint_attention(smpl_feats, part_attention)
        cam_shape_feats = self.keypoint_attention(cam_shape_feats, part_attention)
        # else:
        #     point_local_feat = interpolate(smpl_feats, output['pred_kp2d'])
        #     cam_shape_feats = interpolate(cam_shape_feats, output['pred_kp2d'])
        return point_local_feat, cam_shape_feats
    
    def _pare_get_final_preds(self, pose_feats, cam_shape_feats, init_pose, init_shape, init_cam, iter_now=True):
        pose_feats = pose_feats.unsqueeze(-1)  #
        shape_feats = cam_shape_feats

        shape_feats = torch.flatten(shape_feats, start_dim=1)
        if init_pose.shape[-1] == 6:
            # This means init_pose comes from a previous iteration
            init_pose = init_pose.transpose(2,1).unsqueeze(-1)
        else:
            # This means init pose comes from mean pose
            init_pose = init_pose.reshape(init_pose.shape[0], 6, -1).unsqueeze(-1)

        if self.iterative_regression and iter_now:
            pred_pose = init_pose
            pred_shape = init_shape
            pred_cam = init_cam
            for i in range(self.num_iterations):
                # pose_feats shape: [N, 256, 24, 1]
                # shape_feats shape: [N, 24*64]
                # pose_mlp_inp = self._prepare_pose_mlp_inp(pose_feats, pred_pose, pred_shape, pred_cam)
                # shape_mlp_inp = self._prepare_shape_mlp_inp(shape_feats, pred_pose, pred_shape, pred_cam)
                # residual
                pred_pose = self.pose_mlp(pose_feats) + pred_pose
                pred_cam = self.cam_mlp(shape_feats) + pred_cam
                pred_shape = self.shape_mlp(shape_feats) + pred_shape      
        else:
            pred_pose = self.pose_mlp(pose_feats)
            pred_cam = self.cam_mlp(shape_feats)
            pred_shape = self.shape_mlp(shape_feats)

        # if self.use_mean_camshape:
        #     pred_cam = pred_cam + init_cam
        #     pred_shape = pred_shape + init_shape

        # if self.use_mean_pose:
        #     pred_pose = pred_pose + init_pose
        pred_pose = pred_pose.squeeze(-1).transpose(2, 1) # N, J, 6
        return pred_pose, pred_shape, pred_cam

    def _make_conv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.Conv2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _get_shape_mlp(self, output_size):
        if self.shape_mlp_num_layers == 1:
            return nn.Linear(self.shape_mlp_inp_dim, output_size)

        module_list = []
        for i in range(self.shape_mlp_num_layers):
            if i == 0:
                module_list.append(
                    nn.Linear(self.shape_mlp_inp_dim, self.shape_mlp_hidden_size)
                )
            elif i == self.shape_mlp_num_layers - 1:
                module_list.append(
                    nn.Linear(self.shape_mlp_hidden_size, output_size)
                )
            else:
                module_list.append(
                    nn.Linear(self.shape_mlp_hidden_size, self.shape_mlp_hidden_size)
                )
        return nn.Sequential(*module_list)

    def _get_pose_mlp(self, num_joints, output_size):
        if self.pose_mlp_num_layers == 1:
            return LocallyConnected2d(
                in_channels=self.pose_mlp_inp_dim,
                out_channels=output_size,
                output_size=[num_joints, 1],
                kernel_size=1,
                stride=1,
            )

        module_list = []
        for i in range(self.pose_mlp_num_layers):
            if i == 0:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_inp_dim,
                        out_channels=self.pose_mlp_hidden_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    )
                )
            elif i == self.pose_mlp_num_layers - 1:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_hidden_size,
                        out_channels=output_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    )
                )
            else:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_hidden_size,
                        out_channels=self.pose_mlp_hidden_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    )
                )
        return nn.Sequential(*module_list)
    
    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _prepare_pose_mlp_inp(self, feats, pred_pose, pred_shape, pred_cam):
        # feats shape: [N, 256, J, 1]
        # pose shape: [N, 6, J, 1]
        # cam shape: [N, 3]
        # beta shape: [N, 10]
        batch_size, num_joints = pred_pose.shape[0], pred_pose.shape[2]

        joint_triplets = get_smpl_neighbor_triplets()

        inp_list = []

        for inp_type in self.pose_input_type:
            if inp_type == 'feats':
                # add image features
                inp_list.append(feats)

            if inp_type == 'neighbor_pose_feats':
                # add the image features from neighboring joints
                n_pose_feat = []
                for jt in joint_triplets:
                    n_pose_feat.append(
                        feats[:, :, jt[1:]].reshape(batch_size, -1, 1).unsqueeze(-2)
                    )
                n_pose_feat = torch.cat(n_pose_feat, 2)
                inp_list.append(n_pose_feat)

            if inp_type == 'self_pose':
                # add image features
                inp_list.append(pred_pose)

            if inp_type == 'all_pose':
                # append all of the joint angels
                all_pose = pred_pose.reshape(batch_size, -1, 1)[..., None].repeat(1, 1, num_joints, 1)
                inp_list.append(all_pose)

            if inp_type == 'neighbor_pose':
                # append only the joint angles of neighboring ones
                n_pose = []
                for jt in joint_triplets:
                    n_pose.append(
                        pred_pose[:,:,jt[1:]].reshape(batch_size, -1, 1).unsqueeze(-2)
                    )
                n_pose = torch.cat(n_pose, 2)
                inp_list.append(n_pose)

            if inp_type == 'shape':
                # append shape predictions
                pred_shape = pred_shape[..., None, None].repeat(1, 1, num_joints, 1)
                inp_list.append(pred_shape)

            if inp_type == 'cam':
                # append camera predictions
                pred_cam = pred_cam[..., None, None].repeat(1, 1, num_joints, 1)
                inp_list.append(pred_cam)

        assert len(inp_list) > 0

        # for i,inp in enumerate(inp_list):
        #     print(i, inp.shape)

        return torch.cat(inp_list, 1)

    def _prepare_shape_mlp_inp(self, feats, pred_pose, pred_shape, pred_cam):
        # feats shape: [N, 256, J, 1]
        # pose shape: [N, 6, J, 1]
        # cam shape: [N, 3]
        # beta shape: [N, 10]
        batch_size, num_joints = pred_pose.shape[:2]

        inp_list = []

        for inp_type in self.shape_input_type:
            if inp_type == 'feats':
                # add image features
                inp_list.append(feats)

            if inp_type == 'all_pose':
                # append all of the joint angels
                pred_pose = pred_pose.reshape(batch_size, -1)
                inp_list.append(pred_pose)

            if inp_type == 'shape':
                # append shape predictions
                inp_list.append(pred_shape)

            if inp_type == 'cam':
                # append camera predictions
                inp_list.append(pred_cam)

        assert len(inp_list) > 0

        return torch.cat(inp_list, 1)