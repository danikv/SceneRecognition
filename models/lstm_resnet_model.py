#define the nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging
from graphviz import Digraph

class LSTM_Resnet_Model(nn.Module):
    def __init__(self, num_classes, hidden_dim=8192, num_layers=1):
        super(LSTM_Resnet_Model, self).__init__()
        # self._resnet101 = models.resnet101(pretrained=True)
        # for param in self._resnet101.parameters():
        #     param.requires_grad = False
        # in_features = self._resnet101.fc.in_features
        # self._resnet101.fc = nn.Linear(in_features, 512)
        # self._layer_dim = num_layers
        self._num_classes = num_classes
        self._lstm = nn.LSTM(2048, hidden_dim, num_layers, dropout=0.2, batch_first=True)
        self._hidden_dim = hidden_dim
        self._fc1 = nn.Linear(hidden_dim, 1024)
        self._fc2 = nn.Linear(1024, 512)
        #self._batch = nn.BatchNorm1d(256)
        self._dropout = nn.Dropout(0.2)
        self._fc3 = nn.Linear(512 , 256)
        self._fc4 = nn.Linear(256, num_classes)

    def init_hidden(self, device):
        hidden = torch.zeros(self._layer_dim, 1, self._hidden_dim).to(device)
        cell = torch.zeros(self._layer_dim, 1, self._hidden_dim).to(device)
        self._hidden = hidden, cell
        

    def forward(self, x):
        #x = F.relu(self._resnet101(x)).view(-1, 1, 512)
        seq_len = x.shape[1]
        x, _ = self._lstm(x)
        #self._hidden = hidden
        x = x.view(-1, seq_len, self._hidden_dim)
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._dropout(x)
        x = F.relu(self._fc3(x))
        return self._fc4(x).reshape(seq_len, self._num_classes)


    def train_model(self, training_generator, device, optimizer, criterion, epoch):
        current_loss = 0.0
        num_labels = 0.0
        num_videos = 0.0
        correct_labels = 0.0
        for i, data in enumerate(training_generator, 0):
            # get the inputs; data is a list of [inputs, labels]
            actual_anomylies = 0.0
            anomylies_detected = 0.0
            running_loss = 0.0
            current_labels_video = 0.0
            loss = 0.0
            # zero the parameter gradients
            self.init_hidden(device)
            optimizer.zero_grad()
            for inputs, labels in data.batchiter():
                inputs, labels = inputs.to(device), labels.to(device)
                # forward + backward + optimize
                outputs = self(inputs)
                loss += criterion(outputs, labels)

                # print statistics
                num_labels += labels.shape[0]
                running_loss += loss.item()
                current_loss += loss.item()
                outputs_classification = torch.argmax(outputs, dim=1)
                outputs_classification = torch.argmax(outputs, dim=1)
                actual_anomylies += sum(labels[i] == 1 for i in range(len(labels)))
                anomylies_detected += sum(labels[i] == outputs_classification[i] and labels[i] == 1 for i in range(len(labels)))
                current_labels_video += (labels == outputs_classification).float().sum()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(self._lstm.parameters(), 0.5)
            # for p,n in zip(rnn.parameters(),rnn._all_weights[0]):
            #     if n[:6] == 'weight':
            #         logging.info('===========\ngradient:{}\n----------\n{}'.format(n,p.grad))
            correct_labels += current_labels_video
            logging.info('[%d, %5d] loss: %.3f , anomelies detected : %d , actual anomelies : %d , currect labels : %d' %
                    (epoch + 1, i, loss, anomylies_detected, actual_anomylies, current_labels_video))
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



    def train_model_feature_maps(self, training_generator, device, optimizer, criterion, clip):
        total_loss = 0.0
        num_labels = 0.0
        correct_labels = 0.0
        for i, data in enumerate(training_generator, 0):
            # get the inputs; data is a list of [inputs, labels]
            actual_anomylies = 0.0
            anomylies_detected = 0.0
            # zero the parameter gradients
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
                
            outputs = self(inputs)
            labels = torch.squeeze(labels)
            loss = criterion(outputs, labels)

            # print statistics
            num_labels += labels.shape[0]
            total_loss += loss.item()
            outputs_classification = torch.argmax(outputs, dim=1)
            actual_anomylies += sum(labels[i] == 1 for i in range(len(labels)))
            anomylies_detected += sum(labels[i] == outputs_classification[i] and labels[i] == 1 for i in range(len(labels)))
            correct_labels += (labels == outputs_classification).float().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._lstm.parameters(), clip)
            optimizer.step()
        return total_loss / i, correct_labels / num_labels

    
    def evaluate_model_feature_maps(self, evaluation_generator, device, criterion):
        num_labels = 0.0
        correct_labels = 0.0
        loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(evaluation_generator, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                    
                outputs = self(inputs)
                labels = torch.squeeze(labels)
                loss = criterion(outputs, labels)
                correct_labels += (labels == torch.argmax(outputs, dim=1)).float().sum()
                num_labels += labels.shape[0]
        return loss / i, correct_labels / num_labels