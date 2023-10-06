# Demo

## Flags

- `--vid_file (str)`: Path to input video file or a YouTube link. If you provide a YouTube link it will be downloaded
to a temporary folder and then processed.

- `--cfg (str)`: Path to the configuration file. Use `configs/config.yaml` by default.

- `--ckpt (str)`: Path to the pretrained VPare model.

- `--joint_type (str)`: Output 3D joint format. Use 'spin' by default.

- `--output_folder (str)`: Path to folder to store the VPare predictions and output videos.
 
- `--detector (str), default=yolo`: Defines the type of detector to be used by `bbox` tracking method if enabled. Available options are
`maskrcnn` and `yolo`. `maskrcnn` is more accurate but slower compared to `yolo`.

- `--yolo_img_size (int), default=416`: Input image size of YOLO detector.

- `--tracker_batch_size (int), default=12`: Batch size of the bbox tracker. If you get memory error, you need to reduce it.  

- `--vpare_batch_size (int), default=32`: Batch size of VPARE model (For model without GRU, this param is trivial).

- `--display`: Enable this flag if you want to visualize the output of pose (& shape) estimation.

- `--mesh_render`: This flag enables the final rendering of human meshes. Useful if you only want to visualize the all body reconstruction results.

- `--wireframe`: Enable this if you would like to render wireframe meshes in the above human mesh rendering. 

- `--sideview`: Only available for video rendering with mesh. Render the output meshes from an alternate viewpoint. Default alternate viewpoint is -90 degrees in y axis.
Note that this option doubles the rendering time.

- `--save_obj`: Save output meshes as .obj files.

## Examples
- Run VPare on a video file with CPU and visualize the results with wireframe meshes:
```bash
python demo.py --vid_file sample_video.mp4 --output_folder output/  --display --wireframe --cpu_only
```

- Change the default batch sizes to avoid possible memory errors:
```bash
python demo.py --vid_file sample_video.mp4 --output_folder output/ --tracker_batch_size 2 --vpare_batch_size 64
```

## Output Format

If demo finishes succesfully, it needs to create a file named `[name-of-checkpoint].pkl` in the `--output_folder`.
We can inspect what this file contains by:

```python
>>> import joblib # you may use native pickle here as well

>>> output = joblib.load('output/default_vpare.pkl') 

>>> print(output.keys())  
                                                                                                                                                                                                                                                                                                                                                                                              
dict_keys([1, 2, 3, 4]) # these are the track ids for each subject appearing in the video

>>> for k,v in output[1].items(): print(k,v.shape) 

pred_cam (n_frames, 3)        # weak perspective camera parameters in cropped image space (s,tx,ty)
orig_cam (n_frames, 4)        # weak perspective camera parameters in original image space (sx,sy,tx,ty)
verts (n_frames, 6890, 3)     # SMPL mesh vertices
pose (n_frames, 72)           # SMPL pose parameters
betas (n_frames, 10)          # SMPL body shape parameters
joints3d (n_frames, *49*, 3)  # 3D joints with customized joint order (by default, 49 joints of SPIN model)
joints2d (n_frames, 49, 2)    # 2D joints coordinates w.r.t image, SPIN joint order
bboxes (n_frames, 4)          # bbox detections (cx,cy,w,h)
frame_ids (n_frames,)         # frame ids in which subject with tracking id #1 appears

```
You can find the names & order of 2D/3D joints [here](https://github.com/lisqzqng/VPare/blob/31995dec85f628a25506b003f60b374e9a151ca2/lib/data_utils/kp_utils.py#L200). You can also add additional joint order to [kp_utils.py](https://github.com/lisqzqng/VPare/blob/31995dec85f628a25506b003f60b374e9a151ca2/lib/data_utils/kp_utils.py), use the flag `--joint_type` to customize output 3D joint order.
