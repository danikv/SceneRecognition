#define the nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

class LSTM_Resnet_Model(nn.Module):
    def __init__(self, num_classes, hidden_dim=512, num_layers=3):
        super(LSTM_Resnet_Model, self).__init__()
        self._resnet101 = models.resnet101(pretrained=True)
        for param in self._resnet101.parameters():
            param.requires_grad = False
        in_features = self._resnet101.fc.in_features
        self._resnet101.fc = nn.Linear(in_features, 1024)
        self._layer_dim = num_layers
        self._lstm = nn.LSTM(1024, hidden_dim, num_layers, dropout=0.2)
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
        x = F.relu(self._resnet101(x)).view(-1, 1, 1024)
        x, hidden = self._lstm(x, self._hidden)
        self._hidden = hidden
        x = x.reshape(-1, self._hidden_dim)
        x = F.relu(self._fc2(x))
        x = self._batch(x)
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
                inputs, labels = inputs.to(device), labels.to(device)
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
