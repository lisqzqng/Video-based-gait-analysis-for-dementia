# Instructions & Remarks for 3D human joints batch generation # 
## Precomputed bounding boxes
* Precomputed bboxes are needed to batch generate 3D skeletons from video inputs.

## Basic structure of the output data
* Dictionary of _numpy.ndarray_
* Keys
1. __vid_name__: dim = [total_frame_number,]; name of the video that the frame belongs to, the frames are in temporal order, with fps=20.
2. __joints3D__: dim = [total_frame_number, 25, 3]; the joint hierachy is of format kinectv2; the name and the order of the "kinectv2" 25 joints can be found in "lib/data_utils/kp_utils.py"; please refer to [`joint_image`](kinectv2_25joints.png). 
3. __bbox__: dim = [total_frame_number, 4]; bounding boxes described in step#1, aligned with the number of frames (that are extracted by _ffmpeg_ during 3D joint generation) of the video.

## Run the generation
* Command.
```bash
python batch_generation.py --vid_folder 'folder containing the videos' --bbox_path 'path to to precomputed bbox' --pretrained_file 'path to the pretrained model'
```
* Parameters. The parameters specified at the begining of the code [`batch_generation.py`](../batch_generation.py) can by customized.
