import pytorchvideo.models.resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def make_kinetics_resnet():
  return pytorchvideo.models.resnet.create_resnet(
      input_channel=3, # RGB input from Kinetics
      model_depth=50, # For the tutorial let's just use a 50 layer network
      model_num_class=400,
      norm=nn.BatchNorm3d,
      activation=nn.ReLU,
  )

class Resnet3dLstmModel(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_classes):
        super().__init__()
        self._lstm = nn.LSTM(2048, hidden_dim, num_layers, dropout=0.2, batch_first=True)
        self._fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self._lstm(x)
        return self._fc(x)

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
  def __init__(self, learning_rate, anomaly_classification, hidden_dim):
      super().__init__()
      self._anomaly_classification = anomaly_classification
      if anomaly_classification:
        self._model = Resnet3dLstmModel(hidden_dim, 1, 10)
      else:
        self._model = Resnet3dLstmModel(hidden_dim, 1, 2)
      self._learning_rate = learning_rate

  def forward(self, x):
      return self._model(x)

  def training_step(self, batch, batch_idx):
      y_hat = self._model(batch["video"])
      if self._anomaly_classification:
        y_hat = torch.squeeze(y_hat, 0).reshape(-1, 10)
      else:
        y_hat = torch.squeeze(y_hat, 0).reshape(-1, 2)

      #y_true, _ = torch.max(batch["label"], dim=1)
      y_true = batch['label']
      y_true = torch.squeeze(y_true, 0)

      # Compute cross entropy loss, loss.backwards will be called behind the scenes
      # by PyTorchLightning after being returned from this method.
      loss = F.cross_entropy(y_hat, y_true)
      predictions = torch.argmax(y_hat, dim=1)
      # Log the train loss to Tensorboard
      self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)  
      return { 'loss': loss, 'preds': predictions, 'target': y_true}

  def training_epoch_end(self, outputs):
      if self.current_epoch % 20 == 0:  
        self.epoch_end_metrics(outputs, 'Train', True)
      else:
        self.epoch_end_metrics(outputs, 'Train', False)

  def validation_step(self, batch, batch_idx):
      y_hat = self._model(batch["video"])
      if self._anomaly_classification:
        y_hat = torch.squeeze(y_hat, 0).reshape(-1, 10)
      else:
        y_hat = torch.squeeze(y_hat, 0).reshape(-1, 2)

      #y_true, _ = torch.max(batch["label"], dim=1)
      y_true = batch['label']
      y_true = torch.squeeze(y_true, 0)

      loss = F.cross_entropy(y_hat, y_true)
      predictions = torch.argmax(y_hat, dim=1)
      self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)      
      return { 'loss': loss, 'preds': predictions, 'target': y_true}

  def validation_epoch_end(self, outputs):
      self.epoch_end_metrics(outputs, 'Validation', True)

  def test_step(self, batch, batch_idx):
      y_hat = self._model(batch["video"])
      if self._anomaly_classification:
        y_hat = torch.squeeze(y_hat, 0).reshape(-1, 10)
      else:
        y_hat = torch.squeeze(y_hat, 0).reshape(-1, 2)

      #y_true, _ = torch.max(batch["label"], dim=1)
      y_true = batch['label']
      y_true = torch.squeeze(y_true, 0)

      # Compute cross entropy loss, loss.backwards will be called behind the scenes
      # by PyTorchLightning after being returned from this method.
      loss = F.cross_entropy(y_hat, y_true)
      predictions = torch.argmax(y_hat, dim=1)
      # Log the train loss to Tensorboard
      self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)  
      return { 'loss': loss, 'preds': predictions, 'target': y_true}

  def test_epoch_end(self, outputs):
      self.epoch_end_metrics(outputs, 'Test', True)

  def epoch_end_metrics(self, outputs, mode, log_confusion_matrix):
      preds = torch.cat([tmp['preds'] for tmp in outputs])
      targets = torch.cat([tmp['target'] for tmp in outputs])
      num_classes = 10 if self._anomaly_classification else 2
      #confusion_matrix = torchmetrics.functional.confusion_matrix(preds, targets, num_classes=400)
      accuracy = torchmetrics.functional.accuracy(preds, targets)
      #accuracy_top_5 = torchmetrics.functional.accuracy(preds, targets, top_k=5)
      mAp = torchmetrics.functional.accuracy(preds, targets, average='macro', num_classes=num_classes)
      if log_confusion_matrix:  
        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(num_classes), columns=range(num_classes))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)

        self.logger.experiment.add_figure(f"{mode} Confusion matrix", fig_, self.current_epoch)
    
      self.log(f"{mode} Accuracy per Epoch", accuracy, on_epoch=True)
      #self.log(f"{mode} Accuracy Top 5 per Epoch", accuracy_top_5, on_epoch=True)
      self.log(f"{mode} MAP per Epoch", mAp, on_epoch=True)

  def configure_optimizers(self):
      """
      Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
      usually useful for training video models.
      """
      return torch.optim.Adam(self.parameters(), lr=self._learning_rate)