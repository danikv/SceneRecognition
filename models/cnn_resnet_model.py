#define the nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

class CNN_Resnet_Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self._resnet101 = models.resnet101(pretrained=True)
        for param in self._resnet101.parameters():
            param.requires_grad = False
        in_features = self._resnet101.fc.in_features
        self._resnet101.fc = nn.Linear(in_features, 512)
        self._fc2 = nn.Linear(512, 256)
        self._dropout = nn.Dropout(0.2)
        self._fc3 = nn.Linear(256 , 128)
        self._fc4 = nn.Linear(128, num_classes)
        

    def forward(self, x):
        x = F.relu(self._resnet101(x))
        x = F.relu(self._fc2(x))
        x = self._dropout(x)
        x = F.relu(self._fc3(x))
        return self._fc4(x)


    def train_model(self, training_generator, device, optimizer, criterion, epoch):
        current_loss = 0.0
        num_frames = 0.0
        num_videos = 0.0
        correct_labels = 0.0
        for i, data in enumerate(training_generator, 0):
            running_loss = 0.0
            for inputs, labels in data.batchiter():
                # zero the parameter gradients
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)

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
                    num_labels += labels.shape[0]
                num_videos += 1
        return loss / num_videos, correct_labels / num_labels

