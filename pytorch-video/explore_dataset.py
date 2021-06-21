import argparse
import torch
from UCF_Crime_Data_Module import UCFCrimeDataModule
    
    
parser = argparse.ArgumentParser(description='train a lstm over a videos dataset using grid search')
parser.add_argument('--dataset', help='the dataset folder which contains 2 folders, videos and labels')
parser.add_argument('--clip_duration', type=int, help='size of submpled clip from the original video')
parser.add_argument('--subsampled_frames', type=int, help='size of sub submpled frames from the clip')


args = parser.parse_args()
dataset_folder = args.dataset
clip_duration = args.clip_duration
subsampled_frames = args.subsampled_frames


data_module = UCFCrimeDataModule(dataset_folder, clip_duration, 1, 1, subsampled_frames)

train_dataloader = data_module.train_dataloader()

y_train_classes = [0 for _ in range(11)]

for data in train_dataloader:
    labels = data['label'][0].cpu().numpy()
    for label in labels:
        y_train_classes[label] += 1

print(y_train_classes)


# val_dataloader = data_module.val_dataloader()

# y_val_classes = [0 for _ in range(11)]

# for data in val_dataloader:
#     label = torch.max(data['label'])
#     y_val_classes[label] += 1

# print(y_val_classes)

# test_dataloader = data_module.test_data()

# y_test_classes = [0 for _ in range(11)]

# for data in test_dataloader:
#     label = torch.max(data['label'])
#     y_test_classes[label] += 1

# print(y_test_classes)