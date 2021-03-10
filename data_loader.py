import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
import cv2
import os
import numpy as np
from random import shuffle

def class_labels_into_one_hot(labels):
    label = np.zeros(2, dtype=np.float32)
    if not labels:
        label[0] = 1
    else:
        label[1] = 1
    return label

class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, video_folder, dataset):
        super(MyIterableDataset).__init__()
        self._video_folder = video_folder
        self._dataset = dataset
        self._transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((224,224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    def _generate_labels(self, labels_in_frames, video_frame_index):
        labels = []
        for label in labels_in_frames:
            if int(label[1]) <= video_frame_index <= int(label[2]):
                labels.append(label[0])
        return class_labels_into_one_hot(labels)
        
    def __iter__(self):
        shuffle(self._dataset)
        for video_num, labels in self._dataset:
            cap = cv2.VideoCapture(os.path.join(self._video_folder, 'video_{}.mp4'.format(video_num)))
            index = 0
            video_frames = []
            video_labels = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_labels = self._generate_labels(labels, index)
                index += 1
                video_frames.append(self._transform(frame))
                video_labels.append(frame_labels)
            yield torch.stack(video_frames), np.array(video_labels)