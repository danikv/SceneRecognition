#define the nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

class LSTM_Base_Model(nn.Module):
    def __init__(self, num_classes, hidden_dim=512, num_layers=3):
        super(LSTM_Base_Model, self).__init__()
        self._conv1 = nn.Conv2d(3, 6, 3)
        self._conv2 = nn.Conv2d(6, 16, 3)
        self._conv3 = nn.Conv2d(16, 32, 3)
        self._conv4 = nn.Conv2d(32, 64, 3)
        self._conv5 = nn.Conv2d(64, 128, 3)
        self._conv6 = nn.Conv2d(128, 256, 3)
        self._pool = nn.MaxPool2d(2, 2)
        self._batch_norm1 = nn.BatchNorm2d(256)
        self._layer_dim = num_layers
        self._lstm = nn.LSTM(256 * 3 * 3, hidden_dim, num_layers, dropout=0.2)
        self._hidden_dim = hidden_dim
        self._fc2 = nn.Linear(512, 256)
        self._batch = nn.BatchNorm1d(256)
        self._dropout = nn.Dropout(0.2)
        self._fc3 = nn.Linear(256 , 128)
        self._fc4 = nn.Linear(128, num_classes)
        self._hidden = None


    def init_hidden(self, device):
        hidden = torch.zeros(self._layer_dim, 1, self._hidden_dim).to(device)
        cell = torch.zeros(self._layer_dim, 1, self._hidden_dim).to(device)
        self._hidden = hidden, cell
        

    def forward(self, x):
        x = F.relu(self._pool(self._conv1(x)))
        x = F.relu(self._pool(self._conv2(x)))
        x = F.relu(self._pool(self._conv3(x)))
        x = F.relu(self._pool(self._conv4(x)))
        x = F.relu(self._pool(self._conv5(x)))
        x = F.relu(self._batch_norm1(self._conv6(x)))
        x = x.view(-1, 1, 256 * 3 * 3)
        x, hidden = self._lstm(x, self._hidden)
        self._hidden = hidden
        x = x.reshape(-1, self._hidden_dim)
        x = F.relu(self._fc2(x))
        x = self._dropout(x)
        x = F.relu(self._fc3(x))
        return self._fc4(x)


    def train_model(self, training_generator, device, optimizer, criterion, epoch):
        current_loss = 0.0
        num_labels = 0.0
        num_videos = 0.0
        correct_labels = 0.0
        for i, data in enumerate(training_generator, 0):
            # get the inputs; data is a list of [inputs, labels]
            self.init_hidden(device)
            loss = 0.0
            # zero the parameter gradients
            optimizer.zero_grad()
            for inputs, labels in data.batchiter():
                inputs, labels = inputs.clone().to(device), labels.to(device)
                # forward + backward + optimize
                outputs = self(inputs)
                loss += criterion(outputs, labels)

                # print statistics
                num_labels += labels.shape[0]
                current_loss += loss.item()         
                correct_labels += (labels == torch.argmax(outputs, dim=1)).float().sum()
            loss.backward()
            optimizer.step()
            logging.info('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i, loss))
            num_videos += 1
        return current_loss / num_videos, correct_labels / num_labels

    
    def evaluate_model(self, evaluation_generator, device, criterion):
        num_labels = 0.0
        correct_labels = 0.0
        loss = 0.0
        num_videos = 0.0
        with torch.no_grad():
            for i, data in enumerate(evaluation_generator, 0):
                self.init_hidden(device)
                for inputs, labels in data.batchiter():
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)
                    loss += criterion(outputs, labels)
                    correct_labels += (labels == torch.argmax(outputs, dim=1)).float().sum()
                    num_labels += labels.shape[0]
                num_videos += 1
        return loss / num_videos, correct_labels / num_labels