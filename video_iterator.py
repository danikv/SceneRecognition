import torch
from torchvision import transforms
from moviepy.editor import VideoFileClip
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

class VideoDataset():
    def __init__(self, video_folder, dataset):
        self._video_folder = video_folder
        self._dataset = dataset
        
    def __iter__(self):
        shuffle(self._dataset)
        for video_num, labels in self._dataset:
            print(video_num)
            clip = VideoFileClip(os.path.join(self._video_folder, 'video_{}.mp4'.format(video_num)), audio=False)
            yield VideoIterator(clip, labels)


class VideoIterator():
    def __init__(self, clip, labels):
        self._clip = clip
        self._labels = labels
        self._transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def _generate_labels(self, labels_in_frames, video_frame_index):
        frame_labels = []
        for label in labels_in_frames:
            if int(label[1]) <= video_frame_index <= int(label[2]):
                frame_labels.append(label[0])
        return class_labels_into_one_hot(frame_labels)

    def batchiter(self, batch_size):
        batched_frames = torch.FloatTensor(batch_size, 3, 224, 224)
        batched_labels = np.ndarray(shape=(batch_size, 2), dtype=np.float32)
        for i, (frame, label) in enumerate(self.iter(), 0):
            current_index = i % batch_size
            batched_frames[current_index, :, :, :] = frame
            batched_labels[current_index, :] = label
            if (i + 1) % batch_size == 0:
                yield batched_frames, torch.from_numpy(batched_labels)
        if i % batch_size != 0:
            i = i % batch_size
            print('getting rest of frames : {}'.format(i))
            yield batched_frames.narrow(0, 0, i), torch.from_numpy(batched_labels[:i])
            
    def iter(self):
        for i, frame in enumerate(self._clip.iter_frames(), 0):
            frame_labels = self._generate_labels(self._labels, i)
            yield self._transform(frame), frame_labels
