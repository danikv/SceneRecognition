import numpy as np
import os
import json
import pickle
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
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='train a lstm over a videos dataset')
parser.add_argument('--ephocs', type=int, help='the number of ephocs')
parser.add_argument('--dataset', help='the dataset folder which contains 2 folders, videos and labels')
parser.add_argument('--output_file', help='output file to write loss and evaluation accuracy')

args = parser.parse_args()


dataset_folder = args.dataset
labels_folder = os.path.join(dataset_folder, 'labels')
videos_folder = os.path.join(dataset_folder, 'videos')
ephocs = args.ephocs

classes = ['normal', 'anomaly']


def train_model(model, training_generator, device, optimizer, criterion, batch_size, epoch):
    current_loss = 0.0
    num_frames = 0.0
    for i, data in enumerate(training_generator, 0):
        # get the inputs; data is a list of [inputs, labels]
        batched_inputs, batched_labels = data
        model.init_hidden(device)
        running_loss = 0.0
        for j in range(0, len(batched_inputs[0]), batch_size):
            inputs, labels = batched_inputs[0][j:j + batch_size], batched_labels[0][j:j + batch_size]
            inputs, labels = inputs.to(device), labels.to(device)

            model._hidden[0].detach_()
            model._hidden[1].detach_()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            num_frames += batch_size
            current_loss += loss.item()
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i, running_loss))
    return current_loss / num_frames

def evaluate_model(model, evaluation_generator, device, batch_size):
    num_labels = 0.0
    correct_labels = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(evaluation_generator, 0):
            batched_inputs, batched_labels = data
            model.init_hidden(device)
            for j in range(0, len(batched_inputs[0]), batch_size):
                inputs, labels = batched_inputs[0][j:j + batch_size], batched_labels[0][j:j + batch_size]
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                for i, label in enumerate(labels):
                    if label == outputs[i]:
                        correct_labels += 1
                    num_labels += 1
    return correct_labels / num_labels


train = []
with open(os.path.join(labels_folder, 'train.pkl'), 'rb') as f:
    train = pickle.load(f)

test = []
with open(os.path.join(labels_folder, 'test.pkl'), 'rb') as f:
    test = pickle.load(f)

train, evaluation = train_test_split(train, test_size=0.2)


print(len(train))
print(len(test))
print(len(evaluation))
    

dataset_train = MyIterableDataset(videos_folder, train)
training_generator = torch.utils.data.DataLoader(dataset_train)


dataset_evaluation = MyIterableDataset(videos_folder, evaluation)
evaluation_generator = torch.utils.data.DataLoader(dataset_evaluation)

dataset_test = MyIterableDataset(videos_folder, test)
test_generator = torch.utils.data.DataLoader(dataset_test)


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
evaluation_accuracy = []
loss = []
for epoch in range(ephocs):  # loop over the dataset multiple times
    loss.append(train_model(net, training_generator, device, optimizer, criterion, batch_size, epoch))
    #calculate accuracy over the evaluation dataset
    evaluation_accuracy.append(evaluate_model(net, evaluation_generator, device, batch_size))            

with open(args.output_file, 'w+') as f:
    f.write(str(loss))
    f.write(str(evaluation_accuracy))

print('Finished Training')