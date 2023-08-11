# Gait reconstruction: MAX-GRNet
| Architecture                                                                                               |                                           
|------------------------------------------------------------------------------------------------------------|
| ![Architecture](https://github.com/lisqzqng/Video-based-gait-analysis-for-dementia/blob/master/MAX-GRNet.png)| 

| Qualitative Result                                                                                         | 
|------------------------------------------------------------------------------------------------------------| 
| ![Qualitative Result](https://github.com/lisqzqng/Video-based-gait-analysis-for-dementia/blob/master/sample_with_skeleton.gif)|

## Getting Started
VPare has been implemented and tested on Ubuntu 20.04, GeForce RTX 3090 with python >= 3.7 and cuda >= 11.0. It supports both GPU and CPU inference.

Clone the repo:
```bash
git clone https://github.com/lisqzqng/Video-based-gait-analysis-for-dementia.git
```

Get into the directory VPare/. Install the requirements using `conda`. If you donnot have `conda`, use following commands to install on linux:
```bash
# find the version you want on `https://repo.anaconda.com/miniconda/`
curl -O https://repo.anaconda.com/miniconda/Miniconda-3.7.3-Linux-x86_64.sh
# then follow the prompts. The defaults are generally good.
sh Miniconda-3.7.3-Linux-x86_64.sh
```
## References
This project mainly benefits from the following resources: 
  - Pretrained HMR and some functions are borrowed from [SPIN](https://github.com/nkolot/SPIN).
  - SMPL models and layer is from [SMPL-X model](https://github.com/vchoutas/smplx).
  - Code structure is referred from [VIBE](https://github.com/mkocabas/VIBE).
  - Pretrained Part Attention model and structures are borrowed from [PARE](https://github.com/mkocabas/PARE).
  - Self-attention blocks used in MAX Encoder are hacked from [MAED](https://github.com/ziniuwan/maed)
