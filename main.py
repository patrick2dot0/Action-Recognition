import os
import numpy as np
# third-party library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from load_data import Spatial_Dataset, Load_Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from kLSTM import LSTMCell, separated_LSTM
import pickle
# from utils import *
# from network import *

def labels2cat(list):
    return le.transform(list)

def labels2onehot(list):
    return enc.transform(le.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(y_onehot):
    return le.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


path = "./UCF101/jpegs_256/"    # define UCF-101 spatial data path
action_name_path =  './UCF101actions.pkl';
save_model_path = "./model_ckpt/"

class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        with torch.no_grad():
            for t in range(x_3d.size(1)):
                # CNNs
                x = self.resnet(x_3d[:, t, :, :, :]) # ResNet
                x = x.view(x.size(0), -1)            # flatten output of conv

                # FC layers
                x = self.bn1(self.fc1(x))
                x = F.relu(x)
                x = self.bn2(self.fc2(x))
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_p, training=self.training)
                x = self.fc3(x)

                cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq

def train(log_interval, model, device, train_loader, optimizer, epoch, config):
    # set model as training mode
    spatial_encoder, motion_encoder, MRNNdecoders = model

    spatial_encoder.train()
    motion_encoder.train()
    MRNNdecoders.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, ([X1, X2, X3], y) in enumerate(train_loader):
        # distribute data to device
        X1, X2, X3, y = X1.to(device), X2.to(device), X3.to(device), y.to(device)
        '''(X1, X2, X3) = (spatial, motion x, motion y) images'''
        optimizer.zero_grad()

        X1, X2, X3 = spatial_encoder(X1), motion_encoder(X2), motion_encoder(X3)
        X = torch.cat((X1, X2, X3), dim=2)   # combining all motions

        N_count += X.size(0)

        output, layer_z_T, layer_qz_T, layer_h_T, _ = MRNNdecoders(X, config.tau, is_training=True)
        del X; del X1; del X2; del X3;
        # print("input size", X.size(), "output_size", y_pred.size())

        # computing entropy loss
        entropy = - layer_qz_T * torch.log(layer_qz_T + 1e-20)
        loss2 = torch.mean(torch.sum(entropy, (1, 2)))

        # define total loss
        if config.obj == 'ER':
            loss = F.cross_entropy(output, y) - config.beta * loss2
        elif config.obj == 'VB':
            loss = F.cross_entropy(output, y) - loss2
        elif config.obj == 'MLE':
            loss = F.cross_entropy(output, y)

        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(MRNNdecoders.parameters(), config.clip)
        for p in MRNNdecoders.parameters():
            p.data.add_(-config.lr, p.grad.data)

        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores


def validation(model, device, test_loader, config):
    # set model as testing mode
    spatial_encoder, motion_encoder, MRNNdecoders = model

    spatial_encoder.eval()
    motion_encoder.eval()
    MRNNdecoders.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for [X1, X2, X3], y in test_loader:
            # distribute data to device
            X1, X2, X3, y = X1.to(device), X2.to(device), X3.to(device), y.to(device)
            '''(X1, X2, X3) = (spatial, motion x, motion y) images'''

            X1, X2, X3 = spatial_encoder(X1), motion_encoder(X2),  motion_encoder(X3)
            X = torch.cat((X1, X2, X3), dim=2)   # combining all motions

            output, layer_z_T, layer_qz_T, layer_h_T, _ = MRNNdecoders(X, config.tau, is_training=False)
            del X; del X1; del X2; del X3;
            # print("input size", X.size(), "output_size", y_pred.size())

            # computing entropy loss
            entropy = - layer_qz_T * torch.log(layer_qz_T + 1e-20)
            entropy = entropy.sum(1).sum(1).sum()

            # define total loss
            if config.obj == 'ER':
                loss = F.cross_entropy(output, y) - config.beta * entropy
            elif config.obj == 'VB':
                loss = F.cross_entropy(output, y) - entropy
            elif config.obj == 'MLE':
                loss = F.cross_entropy(output, y)

            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(spatial_encoder.state_dict(), os.path.join(save_model_path, 'spatial_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(motion_encoder.state_dict(), os.path.join(save_model_path, 'motion_encoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(MRNNdecoders.state_dict(), os.path.join(save_model_path, 'MRNNdecoders_epoch{}.pth'.format(epoch + 1)))  # save MRNNdecoders
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    
    return test_loss, test_score


class Config(object):
    init_scale = 0.1
    epochs = 120
    lr = 1e-4
    clip = 0.9         # gradient clip threshold to avoid grdient explosion
    vocab_size = 10000
    tau0 = 5.0  # initial temperature
    anneal_rate = 0.1
    min_temp = 0.1
    beta = 0.001
    log_interval = 100
    obj = 'ER'  # 'MLE','VB','ER'

    # Network Parameters
    # input_size = 28
    hidden_size = 256
    output_size = 101    # UCF101
    # T = 28               # total timesteps
    num_layers = 1
    K = 3

config = Config()


# model learning parameters
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
dropout_p = 0.0
res_size = 224        # input image size of ResNet

# # DecoderRNN architecture
# RNN_hidden_layers = 3
# RNN_hidden_nodes = 512
# RNN_FC_dim = 256

# k = 101        # number of target category
batch_size = 25  # 60
learning_rate = 1e-4
log_interval = 10

# data loading parameters
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)   # load UCF101 actions names

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(y)
# y2 = onehot2labels(y_onehot)

actions = []
fnames = os.listdir(path)

all_names = []
for f in fnames:
    loc1 = f.find('v_')
    loc2 = f.find('_g')
    actions.append(f[(loc1 + 2): loc2])

    all_names.append(path + f)


# list all data files
all_X_list = all_names              # all video file names
all_y_list = labels2cat(actions)    # all video labels

# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)


transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_set, valid_set = Load_Dataset(train_list, train_label, transform=transform), Load_Dataset(test_list, test_label, transform=transform)
train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)


# Create model
spatial_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
motion_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)

# create model
MRNNdecoders = separated_LSTM(cell_class=LSTMCell, input_size=3 * CNN_embed_dim, hidden_size=config.hidden_size,
            output_size=config.output_size, num_layers=config.num_layers, k_cells=config.K, use_bias=True, dropout_prob=0.5).to(device)

# Combine all EncoderCNN + DecoderRNN parameters
cmrnn_params = list(spatial_encoder.fc1.parameters()) + list(spatial_encoder.bn1.parameters()) + \
              list(spatial_encoder.fc2.parameters()) + list(spatial_encoder.bn2.parameters()) + \
              list(spatial_encoder.fc3.parameters()) + \
              list(motion_encoder.fc1.parameters()) + list(motion_encoder.bn1.parameters()) + \
              list(motion_encoder.fc2.parameters()) + list(motion_encoder.bn2.parameters()) + \
              list(motion_encoder.fc3.parameters()) + list(MRNNdecoders.parameters())


optimizer = torch.optim.Adam(cmrnn_params, lr=config.lr)

# Training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []
# Training process
for epoch in range(config.epochs):
    # adjust Gumbel softmax temp
    config.tau = np.maximum(config.tau0 * np.exp(-config.anneal_rate * epoch), config.min_temp)
    
    # training model
    train_losses, train_scores = train(log_interval, [spatial_encoder, motion_encoder, MRNNdecoders], device, train_loader, optimizer, epoch, config)
    # test model
    epoch_test_loss, epoch_test_score = validation([spatial_encoder, motion_encoder, MRNNdecoders], device, valid_loader, config)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./MRNN_epoch_training_losses.npy', A)
    np.save('./MRNN_epoch_training_scores.npy', B)
    np.save('./MRNN_epoch_test_loss.npy', C)
    np.save('./MRNN_epoch_test_score.npy', D)

# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         # test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)         # test loss (on epoch end)
# plt.plot(histories.losses_val)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
title = "./fig_UCF101_CMRNN.png"
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()
