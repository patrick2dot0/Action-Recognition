# Yu-Min-Huang-Action-Recognition
---
## Task Description
In this project, we implement a multistream fusion network  for action recognition (a.k.a. video classification) using UCF101 with PyTorch.

## Setting
- Python 3.6
- torch 0.4.1
- torchvision 0.2.1
- Numpy 1.15.0
- Sklearn 0.19.2
- Matplotlib
- Pandas
- tqdm
- Hardware:
CPU: Intel Core i7-4930K @3.40 GHz
RAM: 64 GB DDR3-1600
GPU: GeForce GTX 1080ti

## Dataset
UCF101 has total 13,320 videos from 101 actions. Videos have various time lengths (frames) and different 2d image size; the shortest is 28 frames.

To avoid painful video preprocessing like frame extraction and conversion such as OpenCV or FFmpeg, here I used a preprocessed dataset from feichtenhofer directly. If you want to convert or extract video frames from scratch, here are some nice tutorials:

https://pythonprogramming.net/loading-video-python-opencv-tutorial/
https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/

