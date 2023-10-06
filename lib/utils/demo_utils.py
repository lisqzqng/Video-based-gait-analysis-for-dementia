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

import os
import cv2
import time
import json
import torch
import subprocess
import numpy as np
import os.path as osp
from pytube import YouTube
from collections import OrderedDict

from lib.utils.smooth_bbox import get_smooth_bbox_params, get_all_bbox_params
from lib.data_utils.img_utils import get_single_image_crop_demo
from lib.utils.geometry import rotation_matrix_to_angle_axis


def preprocess_video(video, joints2d, bboxes, frames, scale=1.0, crop_size=224):
    """
    Read video, do normalize and crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.

    :param video (ndarray): input video
    :param joints2d (ndarray, NxJx3): openpose detections
    :param bboxes (ndarray, Nx5): bbox detections
    :param scale (float): bbox crop scaling factor
    :param crop_size (int): crop width and height
    :return: cropped video, cropped and normalized video, modified bboxes, modified joints2d
    """

    if joints2d is not None:
        bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
        bboxes[:,2:] = 150. / bboxes[:,2:]
        bboxes = np.stack([bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,2]]).T

        video = video[time_pt1:time_pt2]
        joints2d = joints2d[time_pt1:time_pt2]
        frames = frames[time_pt1:time_pt2]

    shape = video.shape

    temp_video = np.zeros((shape[0], crop_size, crop_size, shape[-1]))
    norm_video = torch.zeros(shape[0], shape[-1], crop_size, crop_size)

    for idx in range(video.shape[0]):

        img = video[idx]
        bbox = bboxes[idx]

        j2d = joints2d[idx] if joints2d is not None else None

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=scale,
            crop_size=crop_size)

        if joints2d is not None:
            joints2d[idx] = kp_2d

        temp_video[idx] = raw_img
        norm_video[idx] = norm_img

    temp_video = temp_video.astype(np.uint8)

    return temp_video, norm_video, bboxes, joints2d, frames


def download_youtube_clip(url, download_folder):
    return YouTube(url).streams.first().download(output_path=download_folder)


def trim_videos(filename, start_time, end_time, output_filename):
    command = ['ffmpeg',
               '-i', '"%s"' % filename,
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1',
               '-loglevel', 'panic',
               '"%s"' % output_filename]
    # command = ' '.join(command)
    subprocess.call(command)


def video_to_images(vid_file, img_folder=None, return_info=False, fps=None):
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    if not fps: ### test Diwei, add fps
        command = ['ffmpeg',
                '-i', vid_file,
                '-f', 'image2',
                '-v', 'error',
                f'{img_folder}/%06d.png']
    else:
        command = ['ffmpeg',
                '-i', vid_file,
                '-f', 'image2',
                '-v', 'error',
                '-vf', f'fps={str(fps)}',
                f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder


def download_url(url, outdir):
    print(f'Downloading files from {url}')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)


def download_ckpt(outdir='data/vibe_data', use_3dpw=False):
    os.makedirs(outdir, exist_ok=True)

    if use_3dpw:
        ckpt_file = 'data/vibe_data/vibe_model_w_3dpw.pth.tar'
        url = 'https://www.dropbox.com/s/41ozgqorcp095ja/vibe_model_w_3dpw.pth.tar'
        if not os.path.isfile(ckpt_file):
            download_url(url=url, outdir=outdir)
    else:
        ckpt_file = 'data/vibe_data/vibe_model_wo_3dpw.pth.tar'
        url = 'https://www.dropbox.com/s/amj2p8bmf6g56k6/vibe_model_wo_3dpw.pth.tar'
        if not os.path.isfile(ckpt_file):
            download_url(url=url, outdir=outdir)

    return ckpt_file


def images_to_video(img_folder, output_vid_file, fps=None, start_idx=0):
    os.makedirs(img_folder, exist_ok=True)

    if not fps:
        command = [
            'ffmpeg', '-y', '-threads', '16', '-start_number', f'{start_idx}', '-i', f'{img_folder}/%06d.png', '-profile:v', 'baseline',
            '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
        ]
    else:
        command = ['ffmpeg', '-y', '-threads', '16', '-start_number', f'{start_idx}', '-i', f'{img_folder}/%06d.png', '-profile:v', 'baseline',
            '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', '-vf', f'fps={str(fps)}',output_vid_file,]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam

          
def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size):
    # import IPython; IPython.embed(); exit()
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]

    # unnormalize to crop coords
    keypoints = 0.5 * crop_size * (keypoints + 1.0)

    # rescale to orig img crop
    keypoints *= h[..., None, None] / crop_size

    # transform into original image coords
    keypoints[:,:,0] = (cx - h/2)[..., None] + keypoints[:,:,0]
    keypoints[:,:,1] = (cy - h/2)[..., None] + keypoints[:,:,1]
    return keypoints

          
def prepare_rendering_results(vpare_results, nframes, concat=False):
    assert isinstance(nframes, list), print("Input should be list of valid frames !!")
    if concat:
        frame_results = {nf:{
            'verts':[],'cam':[],'j3d':[], 'j2d':[],
        } for nf in nframes}
    else:
        frame_results = {nf:{} for nf in nframes}
    # convert [person_id][frame_id] to [frme_id][person_id]
    for person_id, person_data in vpare_results.items():
        for idx, frame_id in enumerate(person_data['frame_ids']):
            if concat:
                frame_results[frame_id]['verts'].append(person_data['verts'][idx])
                frame_results[frame_id]['cam'].append(person_data['orig_cam'][idx])
                frame_results[frame_id]['j3d'].append(person_data['joints3d'][idx])
                frame_results[frame_id]['j2d'].append(person_data['joints2d'][idx])
            else:
                frame_results[frame_id][person_id] = {
                    'verts': person_data['verts'][idx],
                    'cam': person_data['orig_cam'][idx],
                    'j3d': person_data['joints3d'][idx],
                    'j2d': person_data['joints2d'][idx],
                }
    # for multi-person case, draw from far to near
    # naive depth ordering based on the scale of the weak perspective camera
    for frame_id, frame_data in frame_results.items():
        if concat:
            for k,v in frame_data.items():
                frame_results[frame_id][k] = np.concatenate(v,axis=0)
        else:
            # sort based on y-scale of the cam in original image coords, Y scale: min-->max
            sort_idx = np.argsort([v['cam'][1] for k,v in frame_data.items()])
            frame_results[frame_id] = OrderedDict(
                {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
            )
    return frame_results

def smooth_tracking(tracking_results, interv=5):
    """
    Smooth out the tracking results to fill the missing frames.Only for bbox.\n
    Return:\n
    `tracking_results`: the smoothed tracking results.\n
    `num_frames`: total number of valid frames.
    """
    assert 'joints2d' not in list(tracking_results.values())[0].keys()
    num_frames = set()
    results = {}
    p_id = 0
    for person_id, result in tracking_results.items():
        assert result['frames'].shape[0] == result['bbox'].shape[0]
        prev_frame = None
        single_dic = {}
        seqs = []
        bbox_params = np.empty(shape=(0, 4), dtype=np.float32)
        for idx, (frame, bbox) in enumerate(zip(result['frames'], result['bbox'])):
            num_frames.add(frame)
            if (prev_frame and (frame-prev_frame)>1) or idx==result['frames'].shape[0]-1:
                if idx==result['frames'].shape[0]-1:
                    if frame-prev_frame==1:
                        seqs.append(idx)
                        bbox_params = np.vstack((bbox_params, bbox))
                    frame+=interv+10
                if (frame-prev_frame)>interv-1:
                    if len(seqs)>MIN_NUM_FRAMES:
                        single_dic['frames'] = result['frames'][seqs]
                        single_dic['bbox'] = bbox_params
                        seqs = []
                        bbox_params = np.empty(shape=(0, 4), dtype=np.float32)
                        results[p_id] = single_dic
                        p_id += 1
                    else:
                        num_to_interpolate = frame-prev_frame-1
                        interp_frame = np.linspace(prev_frame, frame, num_to_interpolate+2).astype(np.int64).tolist()[1:-1]
                        seqs.extend(interp_frame)
                        num_frames.update(interp_frame)
                        prev_bbox = bbox_params[-1]
                        # This will be 4x(n+2)
                        interpolated = np.array(
                            [np.linspace(prev, curr, num_to_interpolate + 2)
                            for prev, curr in zip(prev_bbox, bbox)])
                        bbox_params = np.vstack([bbox_params, interpolated.T[1:-1]])
            seqs.append(idx)
            bbox_params = np.vstack((bbox_params, bbox))
            prev_frame = frame
    del tracking_results
    return results, sorted(list(num_frames))