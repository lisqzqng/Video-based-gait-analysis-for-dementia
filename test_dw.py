# code for tests
import os, sys
sys.path.insert(0, os.getcwd())
import joblib
import os.path as osp
import numpy as np
from lib.utils.demo_utils import prepare_rendering_results

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

MAX = 10
N = 49
# ===== check the ETRI 3D joints output ===== #
base_folder = "./ETRI" # total 764 folder, 774 files with 6 .mat
A = range(1,56)
P = [224,225,226,227,228,229,230]
G = range(1,3)
H = [70,120]
folder_format = "A{:03d}_P{:03d}_G{:03d}_H{:03d}_mp4"
folder_format = osp.join(base_folder, folder_format)
pkl_format = "vpare_woGRU{:d}.pkl"
count = 0
color = ['#00ff00', '#ff0000', '#0000ff', '#0f0f0f', '#f0f0f0']
color2d = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
for a in A:
    for p in P:
        for g in G:
            for h in H:
                folder_name = folder_format.format(a,p,g,h)
                try: assert osp.isdir(folder_name)
                except AssertionError:
                    continue
                else:
                    for n, pkln in enumerate(["vpare_woGRU.pkl"] + [pkl_format.format(x) for x in range(1,MAX)]):
                        fn = osp.join(folder_name, pkln)
                        if not osp.isfile(fn):
                            break
                        else:
                            data = joblib.load(fn)
                            if len(data.keys())>3:
                                # count+=1
                                # print(len(data.keys()))
                                # del data
                                # ===== combine the joint positions in one frame ===== #
                                frames = np.concatenate([v['frame_ids'] for v in data.values()], axis=0)
                                nframes = sorted(list(frames))
                                frame_results = prepare_rendering_results(data, nframes, concat=True)
                                for idx, frame_idx in enumerate(frame_results.keys()):
                                    frame_j3d = frame_results[frame_idx]['j3d']
                                    j2ds = frame_results[frame_idx]['j2d']
                                    if frame_j3d.shape[0]<=2*N:continue
                                    fig = plt.figure("Visualize ETRI")
                                    ax = Axes3D(fig)
                                    """for ind in range(frame_j3d.shape[0]):
                                        joint = frame_j3d[ind]
                                        ax.scatter(joint[0], joint[1], joint[2], c=color[ind//N])
                                        ax.text(joint[0], joint[1], joint[2], str(ind), c=color[ind//N])
                                    RADIUS = 2.0
                                    xroot, yroot, zroot = [0,0,0]
                                    ax.set_xlim3d([(-RADIUS + xroot)/2, (RADIUS + xroot)/2])
                                    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
                                    ax.set_ylim3d([(-RADIUS + yroot)/2, (RADIUS + yroot)/2])
                                    ax.set_xlabel("x")
                                    ax.set_ylabel("y")
                                    ax.set_zlabel("z")
                                    ax.view_init(75, 90) # elevation = 75
                                    plt.show()"""
                                    # 2D joints visualization
                                    img = np.zeros((1920,1088,3)) # base image
                                    for ind in range(j2ds.shape[0]):
                                        joint = j2ds[ind].astype(np.int32)
                                        cv2.circle(img, (joint[0],joint[1]), radius=3, color=color2d[ind//N])
                                        cv2.putText(img, str(ind), (joint[0]+8, joint[1]+8),color=color2d[ind//N],\
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,)
                                    cv2.imshow(f'{folder_name}_{n}',img)
                                    cv2.waitKey()
                                    cv2.destroyAllWindows()

# print(f"Total count: {count}.")