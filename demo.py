import os, sys
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import os.path as osp
import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes as opro

from lib.models.grnet import GRNet
from lib.utils.renderer import Renderer
from lib.utils.vis import draw_3d_skeleton
from lib.utils.geometry import batch_rodrigues
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps

from lib.core.config import parse_args

from lib.utils.demo_utils import (
    download_youtube_clip,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    smooth_tracking,
    download_ckpt,
)

MIN_NUM_FRAMES = 25

def main(args):
    cfg, _ = parse_args(args)
    
    if args.cpu_only:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    video_file = args.vid_file

    if not os.path.isfile(video_file):
        exit(f'Input video \"{video_file}\" does not exist!')

    from datetime import date
    filename = date.today().strftime("%m%d")

    vidtype = 'masked' if 'mask' in os.path.basename(video_file) else 'normal'
    
    if os.path.isfile(video_file):
        output_path = os.path.join(args.output_folder, os.path.basename(video_file.split('.')[0].split('-')[0]), f'{vidtype}-{filename}')
    else:
        sys.exit("invalid input format.")
    os.makedirs(output_path, exist_ok=True)
    # convert video to images sequences
    if args.img_folder and os.path.isdir(args.img_folder):
        image_folder = args.img_folder
        num_frames = len(os.listdir(image_folder))
        img_shape = cv2.imread(os.path.join(image_folder, '000001.png')).shape
    else:
        image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)
    
    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    # ========= Run tracking ========= #
    bbox_scale = 1.0 # change to 1.0 as pare
    trackpath = args.tracking_path
    # run multi object tracker
    if trackpath and os.path.isfile(trackpath):
        tracking_results = joblib.load(trackpath)
        if 0 not in list(tracking_results.keys()):
            tracking_results = {0:tracking_results}
        print(f'Loaded precomputed tracklets from \"{trackpath}\"')
    else:
        mot = MPT(
            device=device,
            batch_size=args.tracker_batch_size,
            display=False,#args.display,
            detector_type=args.detector,
            output_format='dict',
            yolo_img_size=args.yolo_img_size,
        )
        tracking_results = mot(image_folder)

    tracking_results, num_frames_list = smooth_tracking(tracking_results)
    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    # ========= Define Reconstructor model ========= #
    GRNet.is_demo = True
    model = GRNet(
        writer=None,
        seqlen=cfg.DATASET.SEQLEN,
        featcorr=cfg.MODEL.FEAT_CORR,
    ).to(device)
    # ========= Load pretrained weights ========= #
    if args.ckpt and os.path.isfile(args.ckpt): 
        print(f"Loading checkpoint from {args.ckpt}...\n")
        # load pretrained model
        try:
            pretrained_file = args.ckpt
            ckpt = torch.load(pretrained_file)['gen_state_dict']
        except Exception:
            raise ValueError(f"Cannot load pretrinaed model from {pretrained_file}.")
        else:
            model.load_state_dict(ckpt, strict=False)
            print(f'Loaded pretrained weights from \"{pretrained_file}\"')
    model.eval()

    # ========= Run Model on each person ========= #
    print(f'Running Model on each tracklet...')
    grnet_time = time.time()
    grnet_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None

        bboxes = tracking_results[person_id]['bbox']

        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=args.grnet_batch_size, num_workers=16)

        with torch.no_grad():

            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))


            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
            del batch
        
        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        smpl_joints2d = smpl_joints2d.cpu().numpy()

        # Runs 1 Euro Filter to smooth out the results
        if args.smooth:
            min_cutoff = args.smooth_min_cutoff # 0.004
            beta = args.smooth_beta # 1.5
            print(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
            pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                               min_cutoff=min_cutoff, beta=beta)

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        joints2d_img_coord = convert_crop_coords_to_orig_img(
            bbox=bboxes,
            keypoints=smpl_joints2d,
            crop_size=224,
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            #'joints2d': joints2d,
            'joints2d': joints2d_img_coord,
            'bboxes': bboxes,
            'frame_ids': frames,
        }
        
        if args.joint_type!='spin':
            try: 
                output_dict['joints3d'] = convert_kps(pred_joints3d, 'spin', args.joint_type)
                output_dict['joints2d'] = convert_kps(joints2d_img_coord, 'spin', args.joint_type)
            except NameError:
                print(f'Unknown skeleton type: {args.joint_type}.')
        
        grnet_results[person_id] = output_dict

    del model

    end = time.time()
    fps = len(num_frames_list) / (end - grnet_time)

    # get rotation matrix for matplotlib 3D visualization
    vis_orient = np.array([[1,0,0]]) # body orientation for matplotlib 3D axis
    assert pred_joints3d.shape[0] > MIN_NUM_FRAMES
    joints = torch.tensor(pred_joints3d[10].copy())
    h = joints[28,:] - joints[27,:]
    v = joints[40,:] - joints[39,:]
    h = h/torch.linalg.norm(h, keepdim=True)
    v = v/torch.linalg.norm(v, keepdim=True)
    init_orient = torch.cross(h, v).reshape(1,3).cpu().numpy()
    rot_mat_body, _ = opro(vis_orient, init_orient)

    print(f'VIBE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {len(num_frames_list) / total_time:.2f}.')

    assert args.ckpt
    pklname = os.path.basename(args.ckpt).split('.')[0]

    pklname += '.pkl'

    # consider the previous inference results
    idx = 0
    for f in os.listdir(output_path):
        if pklname.split('.')[0] in f and f.endswith('.pkl'): idx+=1

    pklname = pklname if not idx else pklname.split('.')[0]+f"{idx}.pkl"
    pklpath = os.path.join(output_path, pklname)
    print(f'Saving complete output results to \"{pklpath}\".')
    joblib.dump(grnet_results, os.path.join(output_path, f"{pklname}"))

    # ========= Render results as a single video ========= #
    output_img_folder = f'{image_folder}_output'
    os.makedirs(output_img_folder, exist_ok=True)

    print(f'Rendering output video, writing frames to {output_img_folder}.\n(Press any button to quit.)')

    # prepare results for rendering
    frame_results = prepare_rendering_results(grnet_results, num_frames_list)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in grnet_results.keys()}

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    if args.mesh_render:
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)
    else: 
        fig = plt.figure('Video')
        ax_in = fig.add_subplot(1,2,1)
        ax_3d = fig.add_subplot(1,2,2, projection='3d')

    to_break = False
    for frame_idx in tqdm(range(len(image_file_names))):
        if to_break: break
        img_fname = image_file_names[frame_idx]
        img = cv2.imread(img_fname)
        # save the image in case not detection for current frame
        cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

        if args.sideview:
            side_img = np.zeros_like(img)
        
        if not args.mesh_render:
            ax_in.clear()
            ax_in.set_axis_off()
            ax_in.set_title('Input')
            ax_in.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='equal')
            fig.canvas.draw()
            ax_3d.clear()
            ax_3d.set_title('Output')
            ax_3d.view_init(elev=200, azim=-27)
            ax_3d.set_xlim3d([-0.6, 0.6])
            ax_3d.set_ylim3d([-1., 1.])
            ax_3d.set_zlim3d([-1., 1.])
            ax_3d.set_xticks(np.linspace(-0.6, 0.6, 7))
            ax_3d.set_xticklabels([])
            ax_3d.set_yticks(np.linspace(-1., 1., 11))
            ax_3d.set_yticklabels([])
            ax_3d.set_zticks(np.linspace(-1., 1., 11))
            ax_3d.set_zticklabels([])
        
        for person_id, person_data in frame_results[frame_idx].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']
            frame_j3d = np.einsum('ij,kj->ki',rot_mat_body,person_data['j3d']) # TODO currently height is `-z` oriented

            mc = mesh_color[person_id]

            mesh_filename = None

            if args.mesh_render:
                if args.save_obj:
                    render_folder = os.path.join(output_path, 'rendered', f'{person_id:04d}')
                    os.makedirs(render_folder, exist_ok=True)
                    mesh_filename = os.path.join(render_folder, f'{frame_idx:06d}.obj')
                
                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    mesh_filename=mesh_filename,
                )
                
                if args.sideview:
                    side_img = renderer.render(
                        side_img,
                        frame_verts,
                        cam=frame_cam,
                        color=mc,
                        angle=270,
                        axis=[0,1,0],
                    )
                    img = np.concatenate([img, side_img], axis=1)

                cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)
                
            else:
                draw_3d_skeleton(frame_j3d, ax_3d, dataset=args.joint_type)
                assert os.path.isdir(output_img_folder)
                plt.savefig(os.path.join(output_img_folder, f'{frame_idx:06d}.png'))

        if args.display:
            if args.mesh_render:
                cv2.imshow('Video', img)
                if cv2.waitKey(1): # press `q` to quit, & 0xFF == ord('q')
                    to_break = True
                    break
            else:
                plt.draw()
                if plt.waitforbuttonpress(0.01): # press any button to quit TODO
                    to_break = True
                    break

        try:cv2.destroyAllWindows()
        except Exception: plt.close()

    # ========= Save rendered video ========= #
    if args.save_vid:
        save_name = pklname.split('.')[0]+'.mp4'
        save_name = os.path.join(output_path, save_name)
        print(f'Saving result video to {save_name}')
        images_to_video(img_folder=output_img_folder, output_vid_file=save_name)

    shutil.rmtree(output_img_folder)

    if not args.img_folder: shutil.rmtree(image_folder)
    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str, default='',
                        help='input video path or youtube link')

    parser.add_argument('--cfg', type=str, default='configs/config_grnet.yaml',
                        help='comfiguration file for pretrained ckpt.')
                        
    parser.add_argument('--ckpt', type=str, default='',
    			help='path to the pretrained checkpoint.')
    			
    parser.add_argument('--output_folder', type=str, default='output/',
                        help='output folder to write results')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', ],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--grnet_batch_size', type=int, default=450,
                        help='batch size of VIBE')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo (will )')

    parser.add_argument('--mesh_render', action='store_true',
                        help='enable final video rendering of human mesh.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='when render human mesh, add output video with an alternate viewpoint.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')

    parser.add_argument('--smooth_min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. '
                             'Decreasing the minimum cutoff frequency decreases slow speed jitter')

    parser.add_argument('--smooth_beta', type=float, default=0.7,
                        help='one euro filter beta. '
                             'Increasing the speed coefficient(beta) decreases speed lag.')
                             
    parser.add_argument('--tracking_path', type=str, default=None,
                        help='path to the precomputed tracking results. use it to accelerate the global process.')

    parser.add_argument('--img_folder', type=str, default=None)

    parser.add_argument('--joint_type', type=str, default='spin',
                        help='output 3D joint format.')
    
    parser.add_argument('--save_vid', action='store_false',
                        help='save output video to output folder.')

    parser.add_argument('--cpu_only', action='store_true',
                        help='whether to use the original PARE model.')


    args = parser.parse_args()

    main(args)
