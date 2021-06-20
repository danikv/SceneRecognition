import pytorchvideo.models.resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def make_kinetics_resnet():
  return pytorchvideo.models.resnet.create_resnet(
      input_channel=3, # RGB input from Kinetics
      model_depth=50, # For the tutorial let's just use a 50 layer network
      model_num_class=11,
      norm=nn.BatchNorm3d,
      activation=nn.ReLU,
  )

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
  def __init__(self, learning_rate):
      super().__init__()
      self._model = make_kinetics_resnet()
      self._learning_rate = learning_rate

  def forward(self, x):
      return self._model(x)

  def training_step(self, batch, batch_idx):
      # The model expects a video tensor of shape (B, C, T, H, W), which is the
      # format provided by the dataset
      y_hat = self._model(batch["video"])

      # Compute cross entropy loss, loss.backwards will be called behind the scenes
      # by PyTorchLightning after being returned from this method.
      y_true, _ = torch.max(batch["label"], dim=1)
      loss = F.cross_entropy(y_hat, y_true.long())
      predictions = torch.argmax(y_hat, dim=1)
      # Log the train loss to Tensorboard
      self.log('train loss per step', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)  
      return { 'loss': loss, 'preds': predictions, 'target': y_true}

  def train_step_end(self, outputs):
      preds = torch.cat([tmp['preds'] for tmp in outputs])
      targets = torch.cat([tmp['target'] for tmp in outputs])
      confusion_matrix = torchmetrics.functional.confusion_matrix(preds, targets, num_classes=11)
      accuracy = torchmetrics.functional.accuracy(preds, targets)

      df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(11), columns=range(11))
      plt.figure(figsize = (10,7))
      fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
      plt.close(fig_)
        
      self.logger.experiment.add_figure("Train Confusion matrix", fig_, self.current_epoch)
      self.logger.experiment.add_scalar("Train Loss per Epoch", np.mean(outputs['loss'], self.current_epoch))
      self.logger.experiment.add_scalar("Train Accuracy per Epoch", accuracy, self.current_epoch)


  def validation_step(self, batch, batch_idx):
      y_hat = self._model(batch["video"])

      y_true, _ = torch.max(batch["label"], dim=1)
      loss = F.cross_entropy(y_hat, y_true.long())
      predictions = torch.argmax(y_hat, dim=1)
      self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)      
      return { 'loss': loss, 'preds': predictions, 'target': y_true}

  def validation_step_end(self, outputs):
      preds = torch.cat([tmp['preds'] for tmp in outputs])
      targets = torch.cat([tmp['target'] for tmp in outputs])
      confusion_matrix = torchmetrics.functional.confusion_matrix(preds, targets, num_classes=11)
      accuracy = torchmetrics.functional.accuracy(preds, targets)

      df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(11), columns=range(11))
      plt.figure(figsize = (10,7))
      fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
      plt.close(fig_)
        
      self.logger.experiment.add_figure("Validation Confusion matrix", fig_, self.current_epoch)
      self.logger.experiment.add_scalar("Validation Loss per Epoch", np.mean(outputs['loss'], self.current_epoch))
      self.logger.experiment.add_scalar("Validation Accuracy per Epoch", accuracy, self.current_epoch)

  def test_step(self, batch, batch_idx):
      y_hat = self._model(batch["video"])

      y_true, _ = torch.max(batch["label"], dim=1)
      loss = F.cross_entropy(y_hat, y_true.long())
      predictions = torch.argmax(y_hat, dim=1)    
      return { 'loss': loss, 'preds': predictions, 'target': y_true}


  def test_epoch_end(self, outputs):
      preds = torch.cat([tmp['preds'] for tmp in outputs])
      targets = torch.cat([tmp['target'] for tmp in outputs])
      confusion_matrix = torchmetrics.functional.confusion_matrix(preds, targets, num_classes=11)
      accuracy = torchmetrics.functional.accuracy(preds, targets)

      df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(11), columns=range(11))
      plt.figure(figsize = (10,7))
      fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
      plt.close(fig_)
        
      self.logger.experiment.add_figure("Test Confusion matrix", fig_, self.current_epoch)
      self.logger.experiment.add_scalar("Test Loss per Epoch", np.mean(outputs['loss'], self.current_epoch))
      self.logger.experiment.add_scalar("Test Accuracy per Epoch", accuracy, self.current_epoch)

  def configure_optimizers(self):
      """
      Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
      usually useful for training video models.
      """
      return torch.optim.Adam(self.parameters(), lr=self._learning_rate)