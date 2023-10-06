# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import torch
import torch.nn as nn
import numpy as np
import os.path as osp
from smplx import SMPL as _SMPL
from smplx.utils import ModelOutput, SMPLOutput
from smplx.lbs import vertices2joints

from lib.core.config import SMPL_DATA_DIR
from lib.utils.geometry import convert_weak_perspective_to_perspective, perspective_projection

# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27,
    'Left Foot': 10, 'Right Foot': 11,
    'Left Thumb': 35, 'Right Thumb': 40,
}
JOINT_NAMES = [ # final spin joint order
    'OP Nose', # 0
    'OP Neck', # 1
    'OP RShoulder', # 2
    'OP RElbow', # 3
    'OP RWrist', # 4
    'OP LShoulder', # 5
    'OP LElbow', # 6
    'OP LWrist', # 7
    'OP MidHip', # 8
    'OP RHip', # 9
    'OP RKnee', # 10
    'OP RAnkle', # 11
    'OP LHip', # 12
    'OP LKnee', # 13
    'OP LAnkle', # 14
    'OP REye', # 15
    'OP LEye', # 16
    'OP REar', # 17
    'OP LEar', # 18
    'OP LBigToe', # 19
    'OP LSmallToe', # 20
    'OP LHeel', # 21
    'OP RBigToe', # 22
    'OP RSmallToe', # 23
    'OP RHeel', # 24
    'Right Ankle', # 25
    'Right Knee', # 26
    'Right Hip', # 27
    'Left Hip', # 28
    'Left Knee', # 29
    'Left Ankle', # 30
    'Right Wrist', # 31
    'Right Elbow', # 32
    'Right Shoulder', # 33
    'Left Shoulder', # 34
    'Left Elbow', # 35
    'Left Wrist', # 36
    'Neck (LSP)', # 37
    'Top of Head (LSP)', # 38
    'Pelvis (MPII)', # 39
    'Thorax (MPII)', # 40
    'Spine (H36M)', # 41
    'Jaw (H36M)', # 42
    'Head (H36M)', # 43
    'Nose', # 44
    'Left Thumb', # 45
    'Right Thumb', # 46
    'Left Foot', # 47
    'Right Foot', # 48
]

JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}
JOINT_REGRESSOR_TRAIN_EXTRA = osp.join(SMPL_DATA_DIR, 'J_regressor_extra.npy')
SMPL_MEAN_PARAMS = osp.join(SMPL_DATA_DIR, 'smpl_mean_params.npz')
SMPL_MODEL_DIR = SMPL_DATA_DIR
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """
    extra=True
    kinectv2=True
    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        # self.kinectv2 = not self.training
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        if self.extra:
            extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
            if self.kinectv2: # to use the 'spin2' format instead
                left_hands = smpl_output.joints[:,[35,37],:] # left thumb, middle
                right_hands = smpl_output.joints[:,[40,42],:] # right thumb, middle
                thorax = extra_joints[:,JOINT_MAP['Thorax (MPII)']-smpl_output.joints.shape[-2],None]
                joints = torch.cat([smpl_output.joints[:,:24], left_hands, right_hands, thorax], dim=1)
            else:
                joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
                joints = joints[:, self.joint_map, :]
        else:
            joints = smpl_output.joints 
        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            joints=joints,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose)
        return output


def get_smpl_faces():
    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    return smpl.faces

class SMPLHead(nn.Module):
    def __init__(self, 
        focal_length=5000., 
        img_res=224, 
        smpl_model_dir=SMPL_MODEL_DIR,
    ):
        super(SMPLHead, self).__init__()
        self.smpl = SMPL(smpl_model_dir, create_transl=False)
        self.add_module('smpl', self.smpl)
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(self, rotmat, shape, cam=None, normalize_joints2d=False):
        '''
        :param rotmat: rotation in euler angles format (N,J,3,3)
        :param shape: smpl betas
        :param cam: weak perspective camera
        :param normalize_joints2d: bool, normalize joints between -1, 1 if true
        :return: dict with keys 'vertices', 'joints3d', 'joints2d' if cam is True
        '''
        smpl_output = self.smpl(
            betas=shape,
            body_pose=rotmat[:, 1:].contiguous(),
            global_orient=rotmat[:, 0].unsqueeze(1).contiguous(),
            pose2rot=False,
        )

        output = {
            'smpl_vertices': smpl_output.vertices,
            'smpl_joints3d': smpl_output.joints,
        }
        if cam is not None:
            joints3d = smpl_output.joints            
            batch_size = joints3d.shape[0]
            device = joints3d.device
            cam_t = convert_weak_perspective_to_perspective(
                cam,
                focal_length=self.focal_length,
                img_res=self.img_res,
            )
            joints2d = perspective_projection(
                joints3d,
                rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                translation=cam_t,
                focal_length=self.focal_length,
                camera_center=torch.zeros(batch_size, 2, device=device)
            )
            if normalize_joints2d:
                # Normalize keypoints to [-1,1]
                joints2d = joints2d / (self.img_res / 2.)

            output['smpl_joints2d'] = joints2d
            # output['pred_cam_t'] = cam_t

        return output