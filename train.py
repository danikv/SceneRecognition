import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
import cv2
from multiprocessing import Process, Queue
from model import BaseModel
import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import MyIterableDataset


dataset_folder = 'D:\Dataset'
labels_folder = os.path.join(dataset_folder, 'labels')
videos_folder = os.path.join(dataset_folder, 'videos')

classes = ['normal', 'anomaly']


def get_number_from_filename(filename):
    return int(file.split('_')[-1].split('.')[0])

def parse_time(string_time):
    splitted_string = string_time.split(':')
    minutes = int(splitted_string[0])
    seconds = int(splitted_string[1])
    return minutes, seconds


def generate_labels(labels_in_frames, video_frame_index):
    labels = []
    for label in labels_in_frames:
        if int(label[1]) <= video_frame_index <= int(label[2]):
            labels.append(label[0])
    return class_labels_into_one_hot(labels)



#load the preprocessed dataset
with open(os.path.join(labels_folder) + 'unifed_labels2.pkl', 'rb') as f:
    loaded_dataset = pickle.load(f)

print(loaded_dataset)


#filter irellevent categories
for video, atrr in loaded_dataset.items():
    atrr['labels'] = [label for label in atrr['labels'] if label[0] != 'pouring gas' and label[0] != 'gun pointing']
    atrr['labels_in_frames'] = [label for label in atrr['labels_in_frames'] if label[0] != 'pouring gas' and label[0] != 'gun pointing']


#separate data to test and train


train_indexes, test_indexes = train_test_split(range(len(loaded_dataset)), test_size=0.2)

train = {key : value['labels_in_frames'] for key,value in loaded_dataset.items() if key in train_indexes}
test = {key : value['labels_in_frames'] for key,value in loaded_dataset.items() if key in test_indexes}


print(train)
    

dataset = MyIterableDataset(videos_folder, train)

training_generator = torch.utils.data.DataLoader(dataset)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


net = BaseModel(512, 3)
net.to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net.train(True)

#start training
torch.autograd.set_detect_anomaly(True)
batch_size = 256
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(training_generator, 0):
        # get the inputs; data is a list of [inputs, labels]
        batched_inputs, batched_labels = data
        net.init_hidden(device)

        for j in range(0, len(batched_inputs[0]), batch_size):
            inputs, labels = batched_inputs[0][j:j + batch_size], batched_labels[0][j:j + batch_size]
            inputs, labels = inputs.to(device), labels.to(device)

            net._hidden[0].detach_()
            net._hidden[1].detach_()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i, running_loss))
        running_loss = 0.0

print('Finished Training')