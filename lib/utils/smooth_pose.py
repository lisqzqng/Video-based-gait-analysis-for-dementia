# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import numpy as np
import matplotlib.pyplot as plt

from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from lib.utils.one_euro_filter import OneEuroFilter
from lib.utils.geometry import quaternion_to_angle_axis, axisang2quater
from lib.data_utils.kp_utils import convert_kps

colors = ['#00ff00', '#ff00ff', '#0000ff', '#f0f0f0', '#f0f0f0']

def smooth_pose(
    pred_pose:np.ndarray, pred_betas:np.ndarray, 
    min_cutoff=0.004, beta=0.7, device='cpu',kinectv2=False):
    # min_cutoff: Decreasing the minimum cutoff frequency decreases slow speed jitter
    # beta: Increasing the speed coefficient(beta) decreases speed lag.
    if pred_pose.shape[-1]==72:
        qtype = 'axisang'
        pshape = pred_pose.shape
        pred_pose = pred_pose.reshape(pred_betas.shape[0],24,3)
    elif pred_pose.shape[-1]==96:
        qtype = 'quater'
        pshape = pred_pose.shape
        pred_pose = pred_pose.reshape(pred_betas.shape[0],24,4)
    else:
        raise ValueError(f"Invalid pred_pose format: {pred_pose.shape}")

    vis = False
    if vis:
        total = pred_pose.shape[-1]
        for idx in range(pred_pose.shape[-1]):
            ax = plt.subplot(total,1,idx+1)
            ax.plot(range(pred_pose.shape[0]), pred_pose[:,0,idx], c=colors[idx])
        plt.show()

    one_euro_filter = OneEuroFilter(
        np.zeros_like(pred_pose[0]),
        pred_pose[0],
        min_cutoff=min_cutoff,
        beta=beta,
    )

    smpl = SMPL(model_path=SMPL_MODEL_DIR).to(device)
    smpl.kinectv2 = kinectv2

    pred_pose_hat = np.zeros_like(pred_pose)

    # initialize
    pred_pose_hat[0] = pred_pose[0]

    pred_verts_hat = []
    pred_joints3d_hat = []

    if qtype=='axisang':
        _pose = pred_pose[0]
        smpl_output = smpl(
            betas=torch.from_numpy(pred_betas[0]).unsqueeze(0).to(device),
            body_pose=torch.from_numpy(_pose[1:]).unsqueeze(0).to(device),
            global_orient=torch.from_numpy(_pose[0:1]).unsqueeze(0).to(device),
        )
    else:
        _pose = quaternion_to_angle_axis(torch.from_numpy(pred_pose[0].reshape(-1,4)).float())
        smpl_output = smpl(
            betas=torch.from_numpy(pred_betas[0]).unsqueeze(0).to(device),
            body_pose=_pose[1:].unsqueeze(0).to(device),
            global_orient=_pose[0:1].unsqueeze(0).to(device),
        )
    pred_verts_hat.append(smpl_output.vertices.detach().cpu().numpy())
    pred_joints3d_hat.append(smpl_output.joints.detach().cpu().numpy())

    for idx, pose in enumerate(pred_pose[1:]):
        idx += 1

        t = np.ones_like(pose) * idx
        pose = one_euro_filter(t, pose) # smooth w.r.t initial pose
        pred_pose_hat[idx] = pose

        if qtype=='axisang':
            _pose = pose
            smpl_output = smpl(
                betas=torch.from_numpy(pred_betas[0]).unsqueeze(0).to(device),
                body_pose=torch.from_numpy(_pose[1:]).unsqueeze(0).to(device),
                global_orient=torch.from_numpy(_pose[0:1]).unsqueeze(0).to(device),
            )
        else:
            _pose = quaternion_to_angle_axis(torch.from_numpy(pose.reshape(-1,4)).float())
            smpl_output = smpl(
                betas=torch.from_numpy(pred_betas[0]).unsqueeze(0).to(device),
                body_pose=_pose[1:].unsqueeze(0).to(device),
                global_orient=_pose[0:1].unsqueeze(0).to(device),
            )
        pred_verts_hat.append(smpl_output.vertices.detach().cpu().numpy())
        pred_joints3d_hat.append(smpl_output.joints.detach().cpu().numpy())

    if pshape is not None:
        pred_pose_hat = pred_pose_hat.reshape(pshape)
    if kinectv2:
        pred_joints3d = convert_kps(np.vstack(pred_joints3d_hat), src='spin2', dst='kinectv2')
    else:
        pred_joints3d = np.vstack(pred_joints3d_hat)
    return np.vstack(pred_verts_hat), pred_pose_hat, pred_joints3d