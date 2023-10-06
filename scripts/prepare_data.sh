#!/usr/bin/env bash

pip install gdown
mkdir -p data
cd data
gdown "https://drive.google.com/uc?id=13XcBP5tVftsLRRw2dLhmF9JX1iIDyG-5"
unzip vpare_data.zip
rm vpare_data.zip
gdown "https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r"
unzip vibe_data.zip
rm vibe_data.zip
cd ..
mv data/vibe_data/sample_video.mp4 .
mkdir -p $HOME/.torch/models/
mv data/vibe_data/yolov3.weights $HOME/.torch/models/
gdown "https://drive.google.com/uc?id=1Vh9ymxqcJNQNdiT14BTeRwJ8TgzwQyB4"
unzip checkpoint.zip
rm checkpoint.zip
