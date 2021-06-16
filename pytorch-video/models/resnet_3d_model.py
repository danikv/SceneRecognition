import pytorchvideo.models.resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning

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
      self._lr = learning_rate

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

      # Log the train loss to Tensorboard
      self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

      return loss

  def validation_step(self, batch, batch_idx):
      y_hat = self._model(batch["video"])

      y_true, _ = torch.max(batch["label"], dim=1)
      loss = F.cross_entropy(y_hat, y_true.long())
      self.log("val_loss", loss)
      return loss

  def configure_optimizers(self):
      """
      Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
      usually useful for training video models.
      """
      return torch.optim.Adam(self.parameters(), lr=self._lr)