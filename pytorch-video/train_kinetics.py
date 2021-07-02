import os
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import pytorchvideo.data
import torch.utils.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    CenterCrop
)

from models.resnet_3d_model import VideoClassificationLightningModule

class KineticsDataModule(pytorch_lightning.LightningDataModule):

  def __init__(self, data_path, clip_duration, batch_size, num_workers, subsample):
    # Dataset configuration
    super().__init__()
    self._data_path = data_path
    self._clip_duration = clip_duration
    self._batch_size = batch_size
    self._num_workers = num_workers
    self._subsample = subsample

  def train_dataloader(self):
    """
    Create the Kinetics train partition from the list of video labels
    in {self._DATA_PATH}/train
    """
    train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(self._subsample),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            ]
        )
    train_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(self._data_path, "kinetics_train.csv"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._clip_duration),
        decode_audio=False,
        transform=train_transform,
        video_path_prefix=os.path.join(self._data_path, "train"),
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self._batch_size,
        num_workers=self._num_workers,
    )

  def val_dataloader(self):
    """
    Create the Kinetics validation partition from the list of video labels
    in {self._DATA_PATH}/val
    """
    val_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(self._subsample),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    ShortSideScale(256),
                    CenterCrop(244),
                  ]
                ),
              ),
            ]
        )
    val_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(self._data_path, "kinetics_val.csv"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._clip_duration),
        decode_audio=False,
        transform=val_transform,
        video_path_prefix=os.path.join(self._data_path, "val"),
    )
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=self._batch_size,
        num_workers=self._num_workers,
    )

def train(min_ephocs, dataset_folder, batch_size, num_workers, stats_file, clip_duration, model_save_dir, subsampled_frames, learning_rate):
    model = VideoClassificationLightningModule(learning_rate)
    data_module = KineticsDataModule(dataset_folder, clip_duration, batch_size, num_workers, subsampled_frames)
    base_filename = f'resnet-3d-kinetics-{clip_duration}-{subsampled_frames}'
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                        dirpath=f"{model_save_dir}",
                        filename= base_filename + '-{epoch:02d}-{val_loss:.2f}',
                        save_top_k=3,
                        mode='min')
    logger = TensorBoardLogger(stats_file, name=f"resnet-3d-kinetics-{clip_duration}-{subsampled_frames}-{learning_rate}")
    trainer = pytorch_lightning.Trainer(logger=logger, gpus=1, callbacks=[checkpoint_callback],  min_epochs=min_ephocs)
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a lstm over a videos dataset using grid search')
    parser.add_argument('--epochs', type=int, help='the min number of epochs')
    parser.add_argument('--dataset', help='the dataset folder which contains 2 folders, videos and labels')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--num_processes', type=int, help='number of processes to use during trainging')
    parser.add_argument('--model_save_path', help='path for model saving')
    parser.add_argument('--stats_file', help='path for tensor board')
    parser.add_argument('--clip_duration', type=int, help='size of submpled clip from the original video')
    parser.add_argument('--subsampled_frames', type=int, help='size of sub submpled frames from the clip')
    parser.add_argument('--lr', type=float, help="learning rate of the model")

    args = parser.parse_args()

    dataset_folder = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    num_processes = args.num_processes
    stats_file = args.stats_file
    prefix_path = args.model_save_path
    clip_duration = args.clip_duration
    subsampled_frames = args.subsampled_frames
    lr = args.lr
    train(epochs, dataset_folder, batch_size, num_processes, stats_file, clip_duration, prefix_path, subsampled_frames, lr)