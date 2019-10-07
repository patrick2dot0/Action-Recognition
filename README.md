# Yu-Min-Huang-Action-Recognition

Accepted by APSIPA 2019
<Stochastic Fusion for Multi-stream Neural Network in Video Classification>
Author : Yu-Min Huang, Huan-Hsin Tseng, Jen-Tzung Chien
---
## Model
We use CNN (pretrained RESNET 152) to encode images and optical flows.  
And then use RNN (LSTM) decoder with stochastic fusion to generate the final prediction.

+ flow chart  
<img src="figures/flowchart.png" width="30%" height="30%" />

+ CNN encoder  
<img src="figures/CNN.png" width="80%" height="80%" />

+ Multistream fusion  
<img src="figures/multistream_attention.png" width="80%" height="80%" />

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

# Hardware:
- CPU: Intel Core i7-4930K @3.40 GHz
- RAM: 64 GB DDR3-1600
- GPU: GeForce GTX 1080ti

## Dataset
UCF101 has total 13,320 videos from 101 actions. Videos have various time lengths (frames) and different 2d image size; the shortest is 28 frames.

To avoid painful video preprocessing like frame extraction and conversion such as OpenCV or FFmpeg, here I used a preprocessed dataset from feichtenhofer directly. If you want to convert or extract video frames from scratch, here are some nice tutorials:

https://pythonprogramming.net/loading-video-python-opencv-tutorial/
https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/

For convenience, we use preprocessed RGB images and optical flows in feichtenhofer/twostreamfusion
https://github.com/feichtenhofer/twostreamfusion

## Set parameters & path

You should set the path
+ data_path = "./UCF101/jpegs_256/"         # UCF101 video path
+ action_name_path = "./UCF101actions.pkl"
+ save_model_path = "./model_ckpt/"  
+ spatial_data = "./UCF101/jpegs_256/"  
+ motion_x_data = "./UCF101/tvl1_flow/u/"  
+ motion_y_data = "./UCF101/tvl1_flow/v/"  

in main.py and load_data.py

# Results
Accuracy = 85%

Sampling probability of some examples,
+ JumpRope
<img src="figures/probability_transition_JumpRope.png" width="80%" height="80%" />

+ ParallelBars
<img src="figures/probability_transition_ParallelBars.png" width="80%" height="80%" />
