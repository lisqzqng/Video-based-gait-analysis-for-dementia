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

import argparse
import os.path as osp
from yacs.config import CfgNode as CN

# CONSTANTS
# You may modify them at will
SMPL_DATA_DIR = 'data/smpl_data'
GRNET_DATA_DIR = 'data/grnet_data'

# Configuration variables
cfg = CN()

cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 8
cfg.SEED_VALUE = -1

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

# <====== TODO training params
# <====== inference params
cfg.DATASET = CN()
cfg.DATASET.SEQLEN = 100

# <====== model params
cfg.MODEL = CN()
# body-part relationship encoding
cfg.MODEL.PRETRAINED_PARE = osp.join(GRNET_DATA_DIR, 'pare_w_3dpw_checkpoint.ckpt')
cfg.MODEL.BACKBONE_CKPT = osp.join(GRNET_DATA_DIR, 'hrnet_w32.pth.tar')
# whether to use gait feature encoder
cfg.MODEL.USE_GFEAT = True
# feature correction module
cfg.MODEL.FEAT_CORR = CN()
cfg.MODEL.FEAT_CORR.AVG_DIM = 3 # number of outputs for averaged features
cfg.MODEL.FEAT_CORR.ESTIM_PHASE = True
cfg.MODEL.FEAT_CORR.NUM_LAYERS = 1
cfg.MODEL.FEAT_CORR.H_SIZE = 1024
cfg.MODEL.FEAT_CORR.NUM_HEADS = 4
cfg.MODEL.FEAT_CORR.USE_JWFF = False


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, help='cfg file path')

        args = parser.parse_args()
        print(args, end='\n\n')
    
    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    return cfg, cfg_file
