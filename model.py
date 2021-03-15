#define the nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(BaseModel, self).__init__()
        self._resnet101 = models.resnet101(pretrained=True)
        for param in self._resnet101.parameters():
            param.requires_grad = False
        in_features = self._resnet101.fc.in_features
        self._resnet101.fc = nn.Linear(in_features, 1024)
        self._layer_dim = num_layers
        self._lstm = nn.LSTM(1024, hidden_dim, num_layers, dropout=0.2)
        self._hidden_dim = hidden_dim
        self._fc2 = nn.Linear(hidden_dim, 256)
        self._batch = nn.BatchNorm1d(256)
        self._dropout = nn.Dropout(0.2)
        self._fc3 = nn.Linear(256 , 128)
        self._fc4 = nn.Linear(128, 2)
        self._softmax = nn.Softmax()
        self._hidden = None


    def init_hidden(self, device):
        hidden = torch.zeros(self._layer_dim, 1, self._hidden_dim).to(device)
        cell = torch.zeros(self._layer_dim, 1, self._hidden_dim).to(device)
        self._hidden = hidden, cell
        

    def forward(self, x):
        x = self._resnet101(x).view(-1, 1, 1024)
        x, hidden = self._lstm(x, self._hidden)
        self._hidden = hidden
        x = x.reshape(-1, self._hidden_dim)
        x = self._fc2(x)
        x = self._batch(x)
        x = self._dropout(x)
        x = self._fc3(x)
        x = self._fc4(x)
        return self._softmax(x)