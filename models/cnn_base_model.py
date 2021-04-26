#define the nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

class CNN_Base_Model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Base_Model, self).__init__()
        self._conv1 = nn.Conv2d(3, 6, 3)
        self._conv2 = nn.Conv2d(6, 16, 3)
        self._conv3 = nn.Conv2d(16, 32, 3)
        self._conv4 = nn.Conv2d(32, 64, 3)
        self._conv5 = nn.Conv2d(64, 128, 3)
        self._conv6 = nn.Conv2d(128, 256, 3)
        self._pool = nn.MaxPool2d(2, 2)
        self._batch_norm1 = nn.BatchNorm2d(256)
        self._fc1 = nn.Linear(256 * 3 * 3, 1024)
        self._dropout = nn.Dropout(0.2)
        self._fc2 = nn.Linear(1024, 512)
        self._dropout2 = nn.Dropout(0.2)
        self._fc3 = nn.Linear(512 , 256)
        self._fc4 = nn.Linear(256, 128)
        self._fc5 = nn.Linear(128, num_classes)
        

    def forward(self, x):
        x = F.relu(self._pool(self._conv1(x)))
        x = F.relu(self._pool(self._conv2(x)))
        x = F.relu(self._pool(self._conv3(x)))
        x = F.relu(self._pool(self._conv4(x)))
        x = F.relu(self._pool(self._conv5(x)))
        x = F.relu(self._batch_norm1(self._conv6(x)))
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self._fc1(x))
        x = self._dropout(x)
        x = F.relu(self._fc2(x))
        x = self._dropout2(x)
        x = F.relu(self._fc3(x))
        x = F.relu(self._fc4(x))
        return F.relu(self._fc5(x))


    def train_model(self, training_generator, device, optimizer, criterion, epoch):
        current_loss = 0.0
        num_frames = 0.0
        num_videos = 0.0
        correct_labels = 0.0
        for i, data in enumerate(training_generator, 0):
            running_loss = 0.0
            for inputs, labels in data.batchiter():
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                num_frames += labels.shape[0]
                current_loss += loss.item()
                running_loss += loss.item()            
                correct_labels += (labels == torch.argmax(outputs, dim=1)).float().sum()
            logging.info('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i, running_loss))
            num_videos += 1
        return current_loss / num_videos, correct_labels / num_frames


    def evaluate_model(self, evaluation_generator, device, criterion):
        num_labels = 0.0
        correct_labels = 0.0
        loss = 0.0
        num_videos = 0.0
        with torch.no_grad():
            for i, data in enumerate(evaluation_generator, 0):
                for inputs, labels in data.batchiter():
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)
                    loss += criterion(outputs, labels)
                    correct_labels += (labels == torch.argmax(outputs, dim=1)).float().sum()
                    num_labels += len(labels)
                num_videos += 1
        return loss / num_videos, correct_labels / num_labels

