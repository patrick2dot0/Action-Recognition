import torch
import pickle
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import random
import os
import numpy as np
from skimage.transform import rescale, resize

max_frame, img_x, img_y = 240, 90, 120

# For CRNN use. end_extract = max frame to be considered
begin_frame, end_frame, skip_frame = 1, 29, 1

# For transfer learing, image rescaled for the ResNet
img_resize = 224

spatial_data = './UCF101/jpegs_256/'
motion_x_data = './UCF101/tvl1_flow/u/'
motion_y_data = './UCF101/tvl1_flow/v/'

class Load_Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, folders, labels, transform=None):
        "Initialization"
        self.labels = labels
        self.folders = folders
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, data_path, selected_folder, use_transform):
        X = []
        for i in range(begin_frame, end_frame, skip_frame):
            image = Image.open(os.path.join(data_path, selected_folder, 'frame{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load input data
        # files= os.listdir(os.path.join(spatial_data_path, folder))

        X1 = self.read_images(spatial_data, folder, self.transform)    # spatial images
        X2 = np.tile(self.read_images(motion_x_data, folder, self.transform), (1, 3, 1, 1))   # motion x: gray -> GRB
        X3 = np.tile(self.read_images(motion_y_data, folder, self.transform), (1, 3, 1, 1))   # motion y: gray -> GRB

        # labels
        y = torch.from_numpy(np.array(self.labels[index])).type(torch.LongTensor)   # LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return [X1, X2, X3], y

