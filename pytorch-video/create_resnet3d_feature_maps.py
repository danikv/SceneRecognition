from collections import defaultdict
import pytorchvideo.models.resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from UCF_Crime_Data_Module import UCFCrimeDataModule
import argparse
import numpy as np
import os
import pickle

def load_pretrained_model():
  model_name = "slow_r50"
  model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
  model.blocks[-1].proj = nn.Identity()
  return model


parser = argparse.ArgumentParser(description='train a lstm over a videos dataset using grid search')
parser.add_argument('--dataset', help='the dataset folder which contains 2 folders, videos and labels')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--num_processes', type=int, help='number of processes to use during trainging')
parser.add_argument('--clip_duration', type=int, help='size of submpled clip from the original video')
parser.add_argument('--subsampled_frames', type=int, help='size of sub submpled frames from the clip')
parser.add_argument('--output_folder', help='the dataset folder which contains 2 folders, videos and labels')

args = parser.parse_args()

dataset_folder = args.dataset
batch_size = args.batch_size
num_processes = args.num_processes
clip_duration = args.clip_duration
subsampled_frames = args.subsampled_frames
output_folder = args.output_folder

model = load_pretrained_model()

dataset = UCFCrimeDataModule(dataset_folder, clip_duration, batch_size, num_processes, subsampled_frames)
train = dataset.val_dataloader()
current_video = []
new_dataset = defaultdict(list)
last_video_name = ''

for data in train:
    video = data['video']
    names = data['video_name']
    names = [name.split('.')[0] for name in names]
    labels, _ = torch.max(data['label'], dim=1)
    features = model(video)
    for i, name in enumerate(names):
        if last_video_name != '' and last_video_name != name:
            video_features = torch.from_numpy(np.array(current_video))
            torch.save(video_features, os.path.join(output_folder, last_video_name))
            print(last_video_name)
            print(video_features.shape)
            last_video_name = name
            current_video = []
        elif last_video_name == '':
            last_video_name = name
        new_dataset[name].append(labels[i].cpu().detach().clone())
        current_video.append(features[i].cpu().detach().numpy())
    #print(new_dataset[index][0][1].shape)


with open(os.path.join(output_folder, 'val_dataset.pickle'), 'wb') as handle:
    pickle.dump(new_dataset, handle)

