import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from moviepy.editor import VideoFileClip
import os
import numpy as np
from random import shuffle

def class_labels_into_one_hot(labels):
    if not labels:
        return 0
    return 1

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
            print(video_num)
            clip = VideoFileClip(os.path.join(self._video_folder, 'video_{}.mp4'.format(video_num)), audio=False)
            array_size = int(clip.fps * clip.duration) + 1
            video_frames = torch.FloatTensor(array_size, 3, 224, 224)
            video_labels = np.ndarray(shape=(array_size), dtype=np.int64)
            for i, frame in enumerate(clip.iter_frames(), 0):
                frame_labels = self._generate_labels(labels, i)
                video_frames[i, :, :, :] = self._transform(frame)
                video_labels[i] = frame_labels
            clip.close()
            del clip
            yield video_frames, video_labels