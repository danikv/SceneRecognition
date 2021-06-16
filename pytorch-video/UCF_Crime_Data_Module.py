import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
import pandas as pd
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

def normalize_image(x):
    return x / 255.0

class UCFCrimeDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, data_path, clip_duration, batch_size, num_workers):
        # Dataset configuration
        super().__init__()
        self._data_path = data_path
        self._clip_duration = clip_duration
        self._batch_size = batch_size
        self._num_workers = num_workers

    def train_dataloader(self):
        """
        """
        train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(8),
                    Lambda(normalize_image),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            ]
        )
        train_dataset = pytorchvideo.data.UCFCrimeDataset(
            'D:\\Dataset\\ucf_crime.csv',
            'D:\\Dataset\\Videos',
            pytorchvideo.data.make_clip_sampler("uniform", self._clip_duration),
            decode_audio=False,
            transform=train_transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def val_dataloader(self):
        """
        """
        train_dataset = pytorchvideo.data.UCFCrimeDataset(
            'D:\\Dataset\\ucf_crime.csv',
            'D:\\Dataset\\Videos',
            pytorchvideo.data.make_clip_sampler("uniform", self._clip_duration),
            decode_audio=False,
            transform=None,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )