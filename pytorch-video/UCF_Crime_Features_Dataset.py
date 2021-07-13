import torch
from torch.utils.data import Dataset
import pytorch_lightning
import pickle
import os

class UCFCrimeFeaturesDataset(Dataset):
    def __init__(self, annotations_file, features_dir):
        super().__init__()
        with open(annotations_file) as f:
            labels = pickle.load(f)
        self._labels = [(feature_file_name, video_labels) for feature_file_name, video_labels in labels.items()]
        self._features_dir = features_dir

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        feature_file_name, video_labels = self._labels[idx]
        feature_path = os.path.join(self._features_dir, feature_file_name)
        features = torch.load(feature_path)
        return features, video_labels


class UCFCrimeFeatureDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, feature_dir, batch_size, num_workers):
        super().__init__()
        self._feature_dir = feature_dir
        self._batch_size = batch_size
        self._num_workers = num_workers

    def train_dataloader(self):
        train_dataset = UCFCrimeFeaturesDataset(os.path.join(self._feature_dir, "dataset.pickle"), self._feature_dir)
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            pin_memory=True,
            num_workers=self._num_workers,
        )

    def val_dataloader(self):
        val_dataset = UCFCrimeFeaturesDataset(os.path.join(self._feature_dir, "val_dataset.pickle"), self._feature_dir)
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            pin_memory=True,
            num_workers=self._num_workers,
        )

    def test_data(self):
        train_dataset = UCFCrimeFeaturesDataset(os.path.join(self._feature_dir, "test_dataset.pickle"), self._feature_dir)
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            pin_memory=True,
            num_workers=self._num_workers,
        )