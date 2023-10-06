import os, sys
sys.path.insert(0, os.getcwd())
import os.path as osp
import shutil
import time
import joblib
import copy
import scipy.io as sio
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.core.config import update_cfg
from lib.models.vpare import VPARE
from lib.models.grnet import GRNet
from lib.dataset.inference import Inference
from lib.utils.demo_utils import video_to_images
from lib.data_utils.kp_utils import convert_kps
from lib.utils.smooth_pose import smooth_pose
from lib.utils.smooth_bbox import smooth_bbox_params

# =====>> Parameters <<===== #
color2d = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
IMG_W = 1920
IMG_H = 1080
MIN_PIXEL = 500 # threshold criteria for bbox scaling 
BS = 1.8 # scaling ratio for bounding box size less than MIN_PIXEL 
N = 25
M = 3 # minimum of credible joints in a frame
MIN_sdiff = 0.01 # threshold for confidence score difference
MAX_THRESH = 0.3 # threshold for 2D joint confidence score
MIN_FDIFF = 10 # threshold for frame number difference between extracted images and precompted 2D joints
MAX_seqlen = 400 # maximum feasible sequence length for regression in VPare models
# maximum video numbers to be stored in one single output json, 
# i.e. for 100 videos with MAX_VID=50, one will get two output json files.
MAX_VID = 50 

def get_bbox_from_joints2d(kp_2d, smooth=False,threshold=0.1):
    """
    Use K-Medoids Clustering to find the center and the size of the bbox from 2D joints.
    """
    import kmedoids
    from sklearn.metrics.pairwise import euclidean_distances
    assert len(kp_2d.shape)==3 and kp_2d.shape[-2]==N # [seqlen,25,3]
    # =====>> preprocessing
    # replace the less credible joints with the most credible one
    seqlen = kp_2d.shape[0]
    xgrid = np.repeat(np.arange(0,seqlen)[:,None], N, axis=-1)
    _, ygrid = np.meshgrid(np.arange(0,N), np.arange(0,seqlen))
    invalid = kp_2d[:,:,2]<threshold # seqlen, nj
    kp_2d = copy.deepcopy(kp_2d)
    valid = np.argmax(kp_2d[:,:,2], axis=-1)
    ygrid[:] = valid[:,None]
    index = (xgrid.flatten(), ygrid.flatten()) # construct the indexes
    ref = kp_2d[index].reshape(-1,N,3)
    kp_2d[invalid] = ref[invalid]
    # <<< preprocessing DONE.
    ul = np.array([kp_2d[:, :, 0].min(axis=1), kp_2d[:, :, 1].min(axis=1)])  # upper left
    lr = np.array([kp_2d[:, :, 0].max(axis=1), kp_2d[:, :, 1].max(axis=1)])  # lower right
    ul[1] -= (lr[1] - ul[1]) * 0.10  # prevent cutting the head
    w = lr[0] - ul[0]
    h = lr[1] - ul[1]
    # c_x, c_y = ul[0] + w / 2, ul[1] + h / 2
    # =====>> use K-Medoids clustering
    # IMPORTANT!!! assuming that the elderly does not move much throughout the video
    # one bounding box per subject for the entire video
    max_iter = 1000
    eps = 10.0
    # get the bbox centers
    kp = kp_2d.reshape(-1,3).astype(np.float32) # find one center for the entire sequence
    disc = euclidean_distances(kp)
    c = kmedoids.fasterpam(disc, 1, max_iter=max_iter, n_cpu=16)
    c_xy = kp[c.medoids][0,:2]
    # get the bbox sizes
    nw = np.median(w, keepdims=True)
    nh = np.median(h, keepdims=True)
    vis = False
    if vis:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.hist(w, w.shape[0], [0,w.shape[0]], color='r')
        plt.hist(nw, 32, [0,w.shape[0]], color='y')
        plt.ioff()
        plt.show()
    # to keep the aspect ratio
    nw = nh = nh * 1.1
    if nw < MIN_PIXEL:
        nw = nh = nh * BS
    bbox = np.repeat(np.hstack([c_xy, nw, nh])[None,:], seqlen, axis=0)  # shape = (N,4)
    if smooth:
        bbox = smooth_bbox_params(bbox)
    return bbox

def load_openpose_anno(anno_folder='./data/openpose/',):
    "Save a batch of .mat files to a single .json file"
    assert osp.isdir(anno_folder)
    # filter actions with interaction
    interacts = [44,45,46,47,48]
    total, count = 0, 0
    output = {}
    fnames = [osp.join(anno_folder,x) for x in os.listdir(anno_folder)]
    bad_annos = []

    for fn in tqdm(fnames):
        act = int(osp.basename(fn).split('_')[0][1:])
        if act in interacts:
            continue
        op_annos = sio.loadmat(fn)
        joints2d = op_annos['skeleton']
        if joints2d.size==0:
            bad_annos.append(osp.basename(fn))
            continue
        elif not (np.logical_and.reduce((joints2d[:,:,:,2]>0).sum(-1)>M,axis=-1)).sum():
            bad_annos.append(osp.basename(fn))
            continue
        seqlen = joints2d.shape[1]
        vid_name = osp.basename(fn).split('.')[0]
        # assume: at least one skeleton has trustworthy point(s) for each frame
        valid = np.logical_and.reduce(np.logical_or.reduce(joints2d[:,:,2]>MAX_THRESH, axis=-1), axis=-1)
        if valid.sum()==0:
            bad_annos.append(osp.basename(fn))
            continue
        total+=1
        joints2d = joints2d[valid].reshape(-1,seqlen,25,3)
        mask = np.array([True]).astype(np.bool)
        if joints2d.shape[0]>1:
            # compare the confidence score of the bodies
            scores = joints2d[:,:,:,2].mean(-1).mean(-1)
            mask = (scores.max()-scores)<MIN_sdiff

        if mask.sum()>1:
            # if still multiple skeletons, will use the one that takes up more area
            count+=1
        # scale the j2ds to image resolution
        j2ds = joints2d[mask].reshape(-1,seqlen,25,3)
        j2ds[:,:,:,0]*= IMG_W
        j2ds[:,:,:,1]*= IMG_H
        area = 0
        bboxes = None
        out_idx = -1
        for idx, j2d in enumerate(j2ds):
            bbox = get_bbox_from_joints2d(j2d, smooth=False)
            if bbox[0,2]>area:
                area = bbox[0,2]
                bboxes = bbox
                out_idx = idx
        
        output[vid_name] = bboxes
        vis = False
        if vis:
            import cv2
            nmask = mask[np.where(mask)]
            nmask[:] = False
            nmask[out_idx] = True
            for t in tqdm(range(seqlen)):
                img = np.zeros((IMG_H,IMG_W,3)) # base image
                bbox = bboxes[t]
                pt1 = (int(bbox[0]-bbox[2]/2), int(bbox[1]-bbox[2]/2))
                pt2 = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[2]/2))
                cv2.rectangle(img, pt1, pt2, color=(255,255,255),thickness=3)
                for ind in range(j2ds.shape[-2]):
                    joint = j2ds[nmask][0][t][ind]
                    if joint[2]<MAX_THRESH: continue
                    x, y = int(joint[0]), int(joint[1])
                    cv2.circle(img, (x,y), radius=3, color=color2d[ind//N])
                    cv2.putText(img, str(ind%N), (x+8, y+8),color=color2d[ind//N],\
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,)
                cv2.imshow(f'{osp.basename(fn)}',img)
                k = cv2.waitKey()
                if k==ord('q'):break
            cv2.destroyAllWindows()
    print(f"Current with-interation files: {count}/{total}.")
    out_json_path = 'data/coarse_bbox.json'
    joblib.dump(output, out_json_path)
    # save bad annos
    joblib.dump(bad_annos, 'data/sample_wo_joints2D.json')
    return

def prepare_data(
    debug=False,
    fv='data/coarse_bbox.json',
    vid_folder='data/sample_videos',
    outpath="data/outputs.json",
    use_pare=True,
    pretrained_file=None,):
    """
    Create dataset dictionary with precomputed bounding box.\n
    Only process 5 videos when `debug == True`.
    """
    assert osp.isfile(fv)
    annos = joblib.load(fv)
    db = defaultdict(list)
    vidnames = os.listdir(vid_folder)
    # sort the video names
    fun = lambda x: int(x[1:4]+x[6:9]+x[11:14]+x[16:19])
    vidnames = sorted(vidnames, key=fun)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg_file = 'configs/config_grnet.yaml'

    cfg = update_cfg(cfg_file)
    GRNet.is_demo = True
    model = GRNet(
        writer=None,
        seqlen=cfg.DATASET.SEQLEN,
        n_iter=1,
        use_max_encoder=False,
        post_encode=True, # whether use reduced features for temporal encoding
        GRID_ALIGN_Attn=True,
        new_part_attn=True,
    ).to(device)
    # ========= Load pretrained weights ========= #
    if pretrained_file and os.path.isfile(pretrained_file): 
        print(f"Loading checkpoint from {pretrained_file}...\n")
        # load pretrained GRNet model
        try:
            ckpt = torch.load(pretrained_file)['gen_state_dict']
        except Exception:
            raise ValueError(f"Cannot load pretrained model from {pretrained_file}.")
        else:
            model.load_state_dict(ckpt, strict=True)
            print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # =====>>> initialize input
    start = time.time()
    out_ind = 0
    for idx, vid_name in tqdm(enumerate(vidnames)):
        # =====>>> combine annotations and save
        if idx%MAX_VID==0 and idx>0 and (len(vidnames)-idx)>10:
            end = time.time()
            for k,v in db.items():
                if isinstance(v[0], np.ndarray):
                    db[k] = np.concatenate(v, axis=0).astype(np.float32)
                else: 
                    db[k] = np.array(v)
                print(f"{k} shape: {db[k].shape}")
            print(f"=====>>> Generation frame rate: {db['vid_name'].shape[0]/(end-start)}.")
            # =====>>> save annotations
            assert outpath.endswith('.json')
            outfp = outpath[:-5]+f'_{out_ind}.json'                                                                                                                                                     
            joblib.dump(db, outfp)
            print(f"Save database to {outfp}.")
            # update variables
            out_ind += 1
            db = defaultdict(list)
            start = time.time()
        os.system('clear')
        print(u'\u001b[0J'+'='*50+f" process video {idx+1}/{len(vidnames)} "+'='*50)
        if debug and idx>5: break
        if vid_name.split('.')[0] not in list(annos.keys()):
            print(f"Skip video {vid_name}, no precomputed 2D joints!")
            continue
        vid_path = osp.join(vid_folder, vid_name)
        bboxes = annos[vid_name.split('.')[0]]
        frame_num = bboxes.shape[0]
        img_dir = video_to_images(vid_path, return_info=False, fps=20)
        img_files = sorted([x for x in os.listdir(img_dir) if x.endswith('png') or x.endswith('jpg')])
        img_paths = [osp.join(img_dir, x) for x in img_files]
        assert abs(len(img_files)-frame_num)<MIN_FDIFF
        # align frame number
        dframe = len(img_files)-frame_num
        if dframe!=0:
            bboxes = np.repeat(bboxes[0,None,:],len(img_files),axis=0)
            frame_num = len(img_files)
        # =====>>> generate outputs
        outputs = run_grnet_on_frame(model, img_dir, np.array([*range(0,frame_num)]), bboxes, device)
        # get 3D joints features, align frame numbers
        db['vid_name'].extend([vid_name.split('.')[0]]*frame_num)
        db['bbox'].append(bboxes.reshape(frame_num,4))
        db['joints3D'].append(outputs['kp_3d'].cpu().numpy().reshape(frame_num,25,3))
        shutil.rmtree(img_dir)

    # =====>>> combine annotations and save
    if len(db):
        end = time.time()
        for k,v in db.items():
            if isinstance(v[0], np.ndarray):
                db[k] = np.concatenate(v, axis=0).astype(np.float32)
            else: 
                db[k] = np.array(v)
            print(f"{k} shape: {db[k].shape}")
        print(f"=====>>> Generation frame rate: {db['vid_name'].shape[0]/(end-start)}.")
        # =====>>> save annotations
        assert outpath.endswith('.json')
        outfp = outpath[:-5]+f'_{out_ind}.json'                                                                                                                                                     
        joblib.dump(db, outfp)
        print(f"Save database to {outfp}.")

    del model
    return

def run_grnet_on_frame(model, image_folder, frames, bboxes, device)->torch.Tensor:
    "generate MAX-GRNet output"
    dataset = Inference(
        image_folder=image_folder,
        frames=frames,
        bboxes=bboxes,
        joints2d=None,
        scale=1.1, # bbox scale
        use_gait_feat=False,
    )
    bboxes = dataset.bboxes
    frames = dataset.frames
    has_keypoints = False

    dataloader = DataLoader(dataset, batch_size=max(frames.shape[0], MAX_seqlen), num_workers=1)

    with torch.no_grad():

        pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, pred_pareFeat, = [], [], [], [], [], [],

        for batch in dataloader:         
            batch = batch.unsqueeze(0)
            batch = batch.to(device)

            batch_size, seqlen = batch.shape[:2]

            output = model(batch,)
            if isinstance(output, list):
                output = output[-1]
            elif isinstance(output, dict):
                for k,v in output.items():
                    output[k] = v[-1]
            else: raise ValueError(f"Unknown output type {type(output)}")
            joints = output['kp_3d'].detach().cpu().squeeze(0)
            output['kp_3d'] = torch.from_numpy(convert_kps(joints,src='spin2',dst='kinectv2')).float().unsqueeze(0)

            pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
            pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
            start_beta = 75
            pred_betas.append(output['theta'][:, :,start_beta:].reshape(batch_size * seqlen, -1))
            pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))


        pred_cam = torch.cat(pred_cam, dim=0)
        pred_pose = torch.cat(pred_pose, dim=0)
        pred_betas = torch.cat(pred_betas, dim=0)
        pred_joints3d = torch.cat(pred_joints3d, dim=0)

    # =====>> smooth pose using OneEuroFilter
    # pred_pose[:,:3] = torch.tensor([0,0,0]).float().to(device)
    # _, pred_pose, pred_joints3d = smooth_pose(pred_pose.cpu().numpy(), pred_betas.cpu().numpy(),
    #                                         min_cutoff=0.04, beta=0.7, device=device, kinectv2=True)
    del batch

    vis=False
    if vis:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        img = plt.figure('check 3D joints',figsize=(12.,12.))
        ax = Axes3D(img)
        try:
            j3d = output['kp_3d'][0][0].cpu().numpy()
        except:
            j3d = output['kp_3d'][0][0]
        for ind,v in enumerate(j3d):
            ax.scatter(v[0],v[1],v[2],c='#0000ff')
            ax.text(v[0], v[1], v[2], str(ind),c='#0000ff')
        RADIUS = 1.0  # space around the subject
        xroot, yroot, zroot = 0, 0, 0
        ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
        ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
        ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(azim=-89, elev=-73) # image coord view
        plt.show() 

    outputs = {}

    outputs['kp_3d'] = pred_joints3d
    
    return outputs

if __name__=='__main__':
    # load_openpose_anno()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_folder', type=str, default='',
                        help='folder containing videos to process.')
    parser.add_argument('--bbox_path', type=str, default='',
                        help='json file path, the precomputed bbox (.json).')
    parser.add_argument('--outpath', type=str, default=f"data/{time.strftime('%Y%m%d-%H%M%S')}",
                        help='output path to save generated 3D joints.')
    parser.add_argument('--pretrained_file', type=str, default='checkpoint/max-grnet.pth.tar',
                        help='path to the pretrained weights (only for new GRNet).')
    args = parser.parse_args()
    prepare_data(fv=args.bbox_path, vid_folder=args.vid_folder, outpath=args.outpath, \
        pretrained_file=args.pretrained_file,)
