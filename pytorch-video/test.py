import torch
import argparse
from models.resnet_3d_model import VideoClassificationLightningModule
from UCF_Crime_Data_Module import UCFCrimeDataModule
import pytorch_lightning

parser = argparse.ArgumentParser(description='train a lstm over a videos dataset using grid search')
parser.add_argument('--model_path', help='model path')
parser.add_argument('--test_path', help='test set path')
parser.add_argument('--clip_duration', type=int, help='test set path')

args = parser.parse_args()

model_path = args.model_path
data_path = args.test_path
clip_duration = args.clip_duration

model = VideoClassificationLightningModule(0)

data_module = UCFCrimeDataModule(data_path, clip_duration, 1, 1, 1)

trainer = pytorch_lightning.Trainer(gpus=1)

trainer.test(model=model, test_dataloaders=data_module.test_data(), ckpt_path=model_path, verbose=True)
