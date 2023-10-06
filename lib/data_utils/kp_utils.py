# Part of the code is borrowed from https://github.com/mkocabas/VIBE/blob/master/lib/data_utils/kp_utils.py
# Adhere to their licence to use this script

import numpy as np

def get_left_right_index(format_name):
    "get the index of the left joints / right joints respectively."
    joint_names = eval(f'get_{format_name}_joint_names')()
    left, right = [], []
    for idx, name in enumerate(joint_names):
        jn = name.split(' ')[1] if ' ' in name else name
        if jn[0]=='R' or jn[0]=='r': right.append(idx)
        elif jn[0]=='L' or jn[0]=='l': left.append(idx)

    assert len(left)==len(right)
    return left, right

def keypoint_hflip(kp, img_width):
    "Flip a 2D keypoint horizontally around the y-axis in the image." 
    if len(kp.shape) == 2:
        kp[:,0] = (img_width - 1.) - kp[:,0]
    elif len(kp.shape) == 3:
        kp[:, :, 0] = (img_width - 1.) - kp[:, :, 0]
    return kp

def convert_kps(joints2d, src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()

    out_joints2d = np.zeros((joints2d.shape[0], len(dst_names), 3))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[:, idx] = joints2d[:, src_names.index(jn)]

    return out_joints2d

def get_perm_idxs(src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()
    idxs = [src_names.index(h) for h in dst_names if h in src_names]
    return idxs

def get_mpii3d_test_joint_names():
    return [
        'headtop', # 'head_top',
        'neck',
        'rshoulder',# 'right_shoulder',
        'relbow',# 'right_elbow',
        'rwrist',# 'right_wrist',
        'lshoulder',# 'left_shoulder',
        'lelbow', # 'left_elbow',
        'lwrist', # 'left_wrist',
        'rhip', # 'right_hip',
        'rknee', # 'right_knee',
        'rankle',# 'right_ankle',
        'lhip',# 'left_hip',
        'lknee',# 'left_knee',
        'lankle',# 'left_ankle'
        'hip',# 'pelvis',
        'Spine (H36M)',# 'spine',
        'Head (H36M)',# 'head'
    ]

def get_mpii3d_joint_names():
    return [
        'spine3', # 0,
        'spine4', # 1,
        'spine2', # 2,
        'Spine (H36M)', #'spine', # 3,
        'hip', # 'pelvis', # 4,
        'neck', # 5,
        'Head (H36M)', # 'head', # 6,
        "headtop", # 'head_top', # 7,
        'left_clavicle', # 8,
        "lshoulder", # 'left_shoulder', # 9,
        "lelbow", # 'left_elbow',# 10,
        "lwrist", # 'left_wrist',# 11,
        'left_hand',# 12,
        'right_clavicle',# 13,
        'rshoulder',# 'right_shoulder',# 14,
        'relbow',# 'right_elbow',# 15,
        'rwrist',# 'right_wrist',# 16,
        'right_hand',# 17,
        'lhip', # left_hip',# 18,
        'lknee', # 'left_knee',# 19,
        'lankle', #left ankle # 20
        'left_foot', # 21
        'left_toe', # 22
        "rhip", # 'right_hip',# 23
        "rknee", # 'right_knee',# 24
        "rankle", #'right_ankle', # 25
        'right_foot',# 26
        'right_toe' # 27
    ]

def get_insta_joint_names():
    """OpenPose body 25 joints but different order"""
    return [
        'OP RHeel', # 0
        'OP RKnee', # 1
        'rhip (SMPL)', # 2
        'lhip (SMPL)', # 3
        'OP LKnee', # 4
        'OP LHeel', # 5
        'OP RWrist', # 6
        'OP RElbow', # 7
        'OP RShoulder', # 8
        'OP LShoulder', # 9
        'OP LElbow', # 10
        'OP LWrist', # 11
        'OP Neck', # 12
        'headtop', # 13
        'OP Nose', # 14
        'reye', # 15
        'leye', # 16
        'lear', # 17
        'rear', # 18
        'OP LBigToe', # 19
        'OP RBigToe', # 20
        'OP LSmallToe', # 21
        'OP RSmallToe', # 22
        'OP LAnkle', # 23
        'OP RAnkle', # 24
    ]

def get_insta_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [4 , 5],
            [6 , 7],
            [7 , 8],
            [8 , 9],
            [9 ,10],
            [2 , 8],
            [3 , 9],
            [10,11],
            [8 ,12],
            [9 ,12],
            [12,13],
            [12,14],
            [14,15],
            [14,16],
            [15,17],
            [16,18],
            [0 ,20],
            [20,22],
            [5 ,19],
            [19,21],
            [5 ,23],
            [0 ,24],
        ])

def get_staf_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 5],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [8, 12],
            [12, 13],
            [13, 14],
            [0, 15],
            [0, 16],
            [15, 17],
            [16, 18],
            [2, 9],
            [5, 12],
            [1, 19],
            [20, 19],
        ]
    )

def get_staf_joint_names():
    return [
        'OP Nose', # 0,
        'OP Neck', # 1,
        'OP RShoulder', # 2,
        'OP RElbow', # 3,
        'OP RWrist', # 4,
        'OP LShoulder', # 5,
        'OP LElbow', # 6,
        'OP LWrist', # 7,
        'OP MidHip', # 8,
        'OP RHip', # 9,
        'OP RKnee', # 10,
        'OP RAnkle', # 11,
        'OP LHip', # 12,
        'OP LKnee', # 13,
        'OP LAnkle', # 14,
        'OP REye', # 15,
        'OP LEye', # 16,
        'OP REar', # 17,
        'OP LEar', # 18,
        'Neck (LSP)', # 19,
        'Top of Head (LSP)', # 20,
    ]

def get_spin2_joint_names():
    return [
        'hip',              # 0
        'lhip (SMPL)',      # 1
        'rhip (SMPL)',      # 2
        'spine (SMPL)',     # 3
        'lknee',            # 4
        'rknee',            # 5
        'Spine (H36M)',     # 6
        'lankle',           # 7
        'rankle',           # 8
        'spine2',           # 9
        'leftFoot',         # 10
        'rightFoot',        # 11
        'neck',             # 12
        'lcollar',          # 13
        'rcollar',          # 14
        'Head (H36M)',      # 15
        'lshoulder',        # 16
        'rshoulder',        # 17
        'lelbow',           # 18
        'relbow',           # 19
        'lwrist',           # 20
        'rwrist',           # 21
        'leftHand',         # 22
        'rightHand',        # 23
        'leftThumb',        # 24
        'leftHandTip',      # 25
        'rightThumb',       # 26
        'rightHandTip',     # 27
        'thorax',           # 28
    ]

def get_spin_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'rhip (SMPL)',    # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'lhip (SMPL)',    # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'reye',           # 15
        'leye',           # 16
        'rear',           # 17
        'lear',           # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
        'rankle',         # 25
        'rknee',          # 26
        'rhip',           # 27
        'lhip',           # 28
        'lknee',          # 29
        'lankle',         # 30
        'rwrist',         # 31
        'relbow',         # 32
        'rshoulder',      # 33
        'lshoulder',      # 34
        'lelbow',         # 35
        'lwrist',         # 36
        'neck',           # 37
        'headtop',        # 38
        'hip',            # 39 'Pelvis (MPII)', # 39
        'thorax',         # 40 'Thorax (MPII)', # 40
        'Spine (H36M)',   # 41
        'Jaw (H36M)',     # 42
        'Head (H36M)',    # 43
        'nose',           # 44
        'leftThumb',      # 45
        'rightThumb',     # 46
        'leftFoot',       # 47
        'rightFoot',      # 48
    ]

def get_h36m_joint_names():
    # Have checked with MoSH j3d @01-18-2023
    # H36M hips are on the edges, SMPL hips are in the middle
    return [
        'hip',  # 0
        'rhip (H36M)',  # 1
        'rknee',  # 2
        'rankle',  # 3
        'lhip (H36M)',  # 4
        'lknee',  # 5
        'lankle',  # 6
        'Spine (H36M)',  # 7
        'Jaw (H36M)',  # 8
        'Head (H36M)',  # 9
        'headtop',  # 10
        'lshoulder',  # 11
        'lelbow',  # 12
        'lwrist',  # 13
        'rshoulder',  # 14
        'relbow',  # 15
        'rwrist',  # 16
    ]

def get_shcommon_joint_names():
    # Have checked with MoSH j3d @01-18-2023
    return [
        'hip',  # 0
        'rhip',  # 1
        'rknee',  # 2
        'rankle',  # 3
        'lhip',  # 4
        'lknee',  # 5
        'lankle',  # 6
        'Spine (H36M)',  # 7
        'neck',  # 8
        'Jaw (H36M)',  # 9
        'lshoulder',  # 10
        'lelbow',  # 11
        'lwrist',  # 12
        'rshoulder',  # 13
        'relbow',  # 14
        'rwrist',  # 15
    ]

def get_h36m_skeleton(): # left/right alternatively
    return  np.array(
        [
            [0 , 4],[0 , 1],
            [4 , 5],[1 , 2],
            [5 , 6],[2 , 3],
            [0 , 7],[8 , 9],[7 , 8],[9 ,10],
            [8 ,11],[8 ,14],
            [11,12],[14,15],
            [12,13],[15,16]
        ]
    )

def get_spin_skeleton():
    """skeleton with left/right alternative order"""
    return np.array(
        [
            [0 , 1],#left
            [1 , 2 ],[1 , 5 ],            
            [2 , 3 ],[5 , 6 ],            
            [3 , 4 ],[6 , 7 ],            
            [1 , 8 ],[8 , 12],            
            [8 , 9 ],[12, 13],            
            [9 , 10],[13, 14],             
            [10, 11],[0 , 16],
            [0 , 15],[16, 18],
            [15, 17],[21, 20],
            [24, 23],[19, 20],
            [22, 23],[19, 21],
            [22, 24],[14, 21],
            [11, 24],[0 , 38],
        ]
    )

def get_posetrack_joint_names():
    return [
        "nose",
        "neck",
        "headtop",
        "lear",
        "rear",
        "lshoulder",
        "rshoulder",
        "lelbow",
        "relbow",
        "lwrist",
        "rwrist",
        "lhip",
        "rhip",
        "lknee",
        "rknee",
        "lankle",
        "rankle"
    ]

def get_posetrack_original_kp_names():
    return [
        'nose',
        'head_bottom',
        'head_top',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]

def get_pennaction_joint_names():
   return [
       "headtop",   # 0
       "lshoulder", # 1
       "rshoulder", # 2
       "lelbow",    # 3
       "relbow",    # 4
       "lwrist",    # 5
       "rwrist",    # 6
       "lhip" ,     # 7
       "rhip" ,     # 8
       "lknee",     # 9
       "rknee" ,    # 10
       "lankle",    # 11
       "rankle"     # 12
   ]

def get_common_joint_names():
    return [
        "rankle",    # 0  "lankle",    # 0
        "rknee",     # 1  "lknee",     # 1
        "rhip",      # 2  "lhip",      # 2
        "lhip",      # 3  "rhip",      # 3
        "lknee",     # 4  "rknee",     # 4
        "lankle",    # 5  "rankle",    # 5
        "rwrist",    # 6  "lwrist",    # 6
        "relbow",    # 7  "lelbow",    # 7
        "rshoulder", # 8  "lshoulder", # 8
        "lshoulder", # 9  "rshoulder", # 9
        "lelbow",    # 10  "relbow",    # 10
        "lwrist",    # 11  "rwrist",    # 11
        "neck",      # 12  "neck",      # 12
        "headtop",   # 13  "headtop",   # 13
    ]

def get_common_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 2 ],
            [ 8, 9 ],
            [ 9, 3 ],
            [ 2, 3 ],
            [ 8, 12],
            [ 9, 10],
            [12, 9 ],
            [10, 11],
            [12, 13],
        ]
    )

def get_coco_joint_names():
    return [
        "nose",      # 0
        "leye",      # 1
        "reye",      # 2
        "lear",      # 3
        "rear",      # 4
        "lshoulder", # 5
        "rshoulder", # 6
        "lelbow",    # 7
        "relbow",    # 8
        "lwrist",    # 9
        "rwrist",    # 10
        "lhip",      # 11
        "rhip",      # 12
        "lknee",     # 13
        "rknee",     # 14
        "lankle",    # 15
        "rankle",    # 16
    ]

def get_coco_skeleton():
    return np.array(
        [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [ 5, 11],
            [ 6, 12],
            [ 5, 6 ],
            [ 5, 7 ],
            [ 6, 8 ],
            [ 7, 9 ],
            [ 8, 10],
            [ 1, 2 ],
            [ 0, 1 ],
            [ 0, 2 ],
            [ 1, 3 ],
            [ 2, 4 ],
            [ 3, 5 ],
            [ 4, 6 ]
        ]
    )

def get_mpii_joint_names():
    return [
        "rankle",    # 0
        "rknee",     # 1
        "rhip",      # 2
        "lhip",      # 3
        "lknee",     # 4
        "lankle",    # 5
        "hip",       # 6
        "thorax",    # 7
        "neck",      # 8
        "headtop",   # 9
        "rwrist",    # 10
        "relbow",    # 11
        "rshoulder", # 12
        "lshoulder", # 13
        "lelbow",    # 14
        "lwrist",    # 15
    ]

def get_mpii_skeleton():
    # 0  - rankle,
    # 1  - rknee,
    # 2  - rhip,
    # 3  - lhip,
    # 4  - lknee,
    # 5  - lankle,
    # 6  - hip,
    # 7  - thorax,
    # 8  - neck,
    # 9  - headtop,
    # 10 - rwrist,
    # 11 - relbow,
    # 12 - rshoulder,
    # 13 - lshoulder,
    # 14 - lelbow,
    # 15 - lwrist,
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 2, 6 ],
            [ 6, 3 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 9 ],
            [ 7, 12],
            [12, 11],
            [11, 10],
            [ 7, 13],
            [13, 14],
            [14, 15]
        ]
    )

def get_aich_joint_names():
    return [
        "rshoulder", # 0
        "relbow",    # 1
        "rwrist",    # 2
        "lshoulder", # 3
        "lelbow",    # 4
        "lwrist",    # 5
        "rhip",      # 6
        "rknee",     # 7
        "rankle",    # 8
        "lhip",      # 9
        "lknee",     # 10
        "lankle",    # 11
        "headtop",   # 12
        "neck",      # 13
    ]

def get_aich_skeleton():
    # 0  - rshoulder,
    # 1  - relbow,
    # 2  - rwrist,
    # 3  - lshoulder,
    # 4  - lelbow,
    # 5  - lwrist,
    # 6  - rhip,
    # 7  - rknee,
    # 8  - rankle,
    # 9  - lhip,
    # 10 - lknee,
    # 11 - lankle,
    # 12 - headtop,
    # 13 - neck,
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 9, 10],
            [10, 11],
            [12, 13],
            [13, 0 ],
            [13, 3 ],
            [ 0, 6 ],
            [ 3, 9 ]
        ]
    )

def get_3dpw_joint_names():
    return [
        "nose",      # 0
        "thorax",    # 1
        "rshoulder", # 2
        "relbow",    # 3
        "rwrist",    # 4
        "lshoulder", # 5
        "lelbow",    # 6
        "lwrist",    # 7
        "rhip",      # 8
        "rknee",     # 9
        "rankle",    # 10
        "lhip",      # 11
        "lknee",     # 12
        "lankle",    # 13
    ]

def get_3dpw_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 2, 3 ],
            [ 3, 4 ],
            [ 1, 5 ],
            [ 5, 6 ],
            [ 6, 7 ],
            [ 2, 8 ],
            [ 5, 11],
            [ 8, 11],
            [ 8, 9 ],
            [ 9, 10],
            [11, 12],
            [12, 13]
        ]
    )

def get_smplcoco_joint_names():
    return [
        "rankle",    # 0
        "rknee",     # 1
        "rhip",      # 2
        "lhip",      # 3
        "lknee",     # 4
        "lankle",    # 5
        "rwrist",    # 6
        "relbow",    # 7
        "rshoulder", # 8
        "lshoulder", # 9
        "lelbow",    # 10
        "lwrist",    # 11
        "neck",      # 12
        "headtop",   # 13
        "nose",      # 14
        "leye",      # 15
        "reye",      # 16
        "lear",      # 17
        "rear",      # 18
    ]

def get_smplcoco_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 12],
            [12, 9 ],
            [ 9, 10],
            [10, 11],
            [12, 13],
            [14, 15],
            [15, 17],
            [16, 18],
            [14, 16],
            [ 8, 2 ],
            [ 9, 3 ],
            [ 2, 3 ],
        ]
    )

def get_smpl_joint_names():
    return [
        'hips',            # 0
        'leftUpLeg',       # 1
        'rightUpLeg',      # 2
        'spine',           # 3
        'leftLeg',         # 4
        'rightLeg',        # 5
        'spine1',          # 6
        'leftFoot',        # 7
        'rightFoot',       # 8
        'spine2',          # 9
        'leftToeBase',     # 10
        'rightToeBase',    # 11
        'neck',            # 12
        'leftShoulder',    # 13
        'rightShoulder',   # 14
        'head',            # 15
        'leftArm',         # 16
        'rightArm',        # 17
        'leftForeArm',     # 18
        'rightForeArm',    # 19
        'leftHand',        # 20
        'rightHand',       # 21
        'leftHandIndex1',  # 22
        'rightHandIndex1', # 23
    ]

def get_smpl_skeleton():
    # left/right alternatively
    return np.array(
        [
            [ 0, 1 ],#left
            [ 0, 2 ],
            [ 0, 3 ],
            [ 2, 5 ],
            [ 1, 4 ],
            [ 3, 6 ],
            [ 4, 7 ], 
            [ 5, 8 ],
            [ 6, 9 ],
            [ 8, 11],
            [ 7, 10],
            [ 9, 13],
            [ 9, 12],
            [ 9, 14],
            [12, 15],
            [14, 17],
            [13, 16],
            [17, 19],
            [16, 18],
            [19, 21],
            [18, 20],
            [21, 23],
            [20, 22],
        ]
    )

def get_smpl2_joint_names():
    return [
        'hip',              # 0
        'lhip (SMPL)',      # 1
        'rhip (SMPL)',      # 2
        'spine (SMPL)',     # 3
        'lknee',            # 4
        'rknee',            # 5
        'Spine (H36M)',     # 6
        'lankle',           # 7
        'rankle',           # 8
        'spine2',           # 9
        'leftFoot',         # 10
        'rightFoot',        # 11
        'neck',             # 12
        'lcollar',          # 13
        'rcollar',          # 14
        'Jaw (H36M)',      # 15
        'lshoulder',        # 16
        'rshoulder',        # 17
        'lelbow',           # 18
        'relbow',           # 19
        'lwrist',           # 20
        'rwrist',           # 21
        'leftHand',         # 22
        'rightHand',        # 23
    ]

def get_cmu21_joint_names():
    return [
        'hip',              # 0
        'rhip',             # 1
        'rknee',            # 2
        'rankle',           # 3
        'rightFoot',        # 4
        'lhip',             # 5
        'lknee',            # 6
        'lankle',           # 7
        'leftFoot',         # 8
        'thorax',           # 9
        'Spine (H36M)',     # 10
        'neck',             # 11
        'Jaw (H36M)',       # 12
        'rshoulder',        # 13
        'relbow',           # 14
        'rwrist',           # 15
        'rightHand',        # 16
        'lshoulder',        # 17
        'lelbow',           # 18
        'lwrist',           # 19
        'leftHand',         # 20
    ]

def get_h36m32_joint_names():
    """ Joint order checked with world coord in raw data @2023-01-17\n
    only 25 among the 32 joints have valid values"""
    return [
        'hip',              # 0
        'rhip',             # 1
        'rknee',            # 2
        'rankle',           # 3
        'rightFoot',        # 4
        'rightToe',         # 5
        'lhip',             # 6
        'lknee',            # 7
        'lankle',           # 8
        'leftFoot',         # 9
        'leftToe',          # 10
        'spine',            # 11
        'Spine (H36M)',     # 12
        'neck',             # 13
        'Jaw (H36M)',       # 14
        'Head (H36M)',      # 15
        'headtop (H36M)',   # 16 Invalid in raw dara
        'lshoulder',        # 17
        'lelbow',           # 18
        'lwrist',           # 19
        'leftHand',         # 20
        'leftThumb',        # 21
        'leftHandTip',      # 22
        'leftHand2',        # 23
        'necklow2',         # 24
        'rshoulder',        # 25
        'relbow',           # 26
        'rwrist',           # 27
        'rightHand',        # 28
        'rightThumb',       # 29
        'rightHandTip',     # 30
        'rightHand2',       # 31
    ]
    
def get_OP21a_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'Jaw (H36M)',     # 19
        'headtop',        # 20
    ]

def get_OP21a_skeleton():
    # left/right alternatively
    return np.array(
        [
            [ 1, 19],#left
            [ 0, 19],
            [ 1, 8 ],
            [ 0, 20],
            [1 , 5 ],[1 , 2 ],            
            [5 , 6 ],[2 , 3 ],            
            [6 , 7 ],[3 , 4 ],            
            [8 , 12],[1 , 8 ],            
            [12, 13],[8 , 9 ],            
            [13, 14],[9 , 10],             
            [0 , 16],[10, 11],
            [16, 18],[0 , 15],
        ]
    )
def get_kinectv2_joint_names():
    return [
        'hip',             # 0
        'Spine (H36M)',    # 1
        'neck',            # 2
        'Head (H36M)',     # 3
        'lshoulder',       # 4
        'lelbow',          # 5
        'lwrist',          # 6
        'leftHand',        # 7
        'rshoulder',       # 8
        'relbow',          # 9
        'rwrist',          # 10
        'rightHand',       # 11
        'lhip (SMPL)',     # 12
        'lknee',           # 13
        'lankle',          # 14
        'leftFoot',        # 15
        'rhip (SMPL)',     # 16
        'rknee',           # 17
        'rankle',          # 18
        'rightFoot',       # 19
        'thorax',          # 20 spine shoulder
        'leftHandTip',     # 21
        'leftThumb',       # 22
        'rightHandTip',    # 23
        'rightThumb',      # 24
    ]

def get_kinectv2_skeleton():
    # left/right alternatively
    return  np.array(
        [
            [0 , 1],[20, 2],[1 ,20],[2 , 3], # trunk
            [20, 4],[20, 8],[4 , 5],[8 , 9],[5 , 6],[9 ,10], # upper body
            [6 , 7],[10,11],[7 ,21],[11,23],[6 ,22],[10,24], # hands
            [0 ,12],[0 ,16],[12,13],[16,17],[13,14],[17,18], # lower body
            [14,15],[18,19], # feet
        ]
    )