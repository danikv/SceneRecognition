from typing import Callable, Dict, List, Optional, Tuple
import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
import numpy as np
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    UniformTemporalSubsampleOverMultipleKeys
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

def normalize_image(x):
    return x / 255.0


class RemoveTrainingExamplesByProbabilityAndCondition(torch.nn.Module):
    """
    subsample frames as in UniformTemporalSubsample but also uses the same indices for the labels.
    usefull for subsampling frames where the labels are at frame level.
    """

    def __init__(self, probability: float, condition):
        super().__init__()
        self._probability = probability
        self._condition = condition

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (Dict[str, torch.Tensor]): video clip dict.
        """
        samples = [0, 1]
        if self._condition(x):
          if np.random.choice(samples, p=[self._probability, 1 - self._probability]) == 0:
            return None
        return x


class UCFCrimeDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, data_path, clip_duration, batch_size, num_workers, subsample):
        # Dataset configuration
        super().__init__()
        self._data_path = data_path
        self._clip_duration = clip_duration
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._subsample = subsample
        self._fps = 30

    def train_dataloader(self):
        """
        """
        train_transform = Compose(
            [
            UniformTemporalSubsampleOverMultipleKeys(self._subsample, "video", "label"),
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    Lambda(normalize_image),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            RemoveTrainingExamplesByProbabilityAndCondition(0.5, lambda x: np.max(x['label']) == 0),
            ]
        )
        train_dataset = pytorchvideo.data.UCFCrimeDataset(
            os.path.join(self._data_path, 'ucf_crime_train.csv'),
            os.path.join(self._data_path, 'Videos'),
            pytorchvideo.data.make_clip_sampler("uniform", self._clip_duration),
            decode_audio=False,
            transform=train_transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            pin_memory=True,
            num_workers=self._num_workers,
        )

    def val_dataloader(self):
        """
        """
        val_transform = Compose(
            [
            UniformTemporalSubsampleOverMultipleKeys(self._subsample, "video", "label"),
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    #UniformTemporalSubsample(self._clip_duration * self._fps),
                    Lambda(normalize_image),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                  ]
                ),
              ),
            ]
        )
        val_dataset = pytorchvideo.data.UCFCrimeDataset(
            os.path.join(self._data_path, 'ucf_crime_val.csv'),
            os.path.join(self._data_path, 'Videos'),
            pytorchvideo.data.make_clip_sampler("uniform", self._clip_duration),
            decode_audio=False,
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._batch_size,
            pin_memory=True,
            num_workers=self._num_workers,
        )

    def test_data(self):
        test_transform = Compose(
            [
            UniformTemporalSubsampleOverMultipleKeys(self._subsample, "video", "label"),
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    #UniformTemporalSubsample(self._clip_duration * self._fps),
                    Lambda(normalize_image),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                  ]
                ),
              ),
            ]
        )
        val_dataset = pytorchvideo.data.UCFCrimeDataset(
            os.path.join(self._data_path, 'ucf_crime_test.csv'),
            os.path.join(self._data_path, 'Videos'),
            pytorchvideo.data.make_clip_sampler("uniform", self._clip_duration),
            decode_audio=False,
            transform=test_transform,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._batch_size,
            pin_memory=True,
            num_workers=self._num_workers,
        )