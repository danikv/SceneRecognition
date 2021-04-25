import torch
from torchvision import transforms
from moviepy.editor import VideoFileClip
import os
import numpy as np
from random import shuffle
from multiprocessing import Pool, Queue
import cv2
from functools import partial
import logging

transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def class_labels_into_one_hot(labels):
    if not labels:
        return 0
    return 1

class ImageSequenceDataset():
    def __init__(self, video_folder, dataset, num_processes, batch_size):
        self._video_folder = video_folder
        self._dataset = dataset
        self._num_processes = num_processes
        self._batch_size = batch_size
        
    def __iter__(self):
        shuffle(self._dataset)
        pool = Pool(self._num_processes)
        for video_num, labels in self._dataset:
            image_sequence_folder = os.path.join(self._video_folder, 'video_{}'.format(video_num))
            logging.debug('starting video {}, num frames : {}'.format(video_num, labels['num_frames']))
            yield ImageSequenceIterator(image_sequence_folder, labels, self._num_processes, pool, self._batch_size)

def generate_labels(labels_in_frames, video_frame_index):
    frame_labels = []
    for label in labels_in_frames:
        if int(label[1]) <= video_frame_index <= int(label[2]):
            frame_labels.append(label[0])
    return class_labels_into_one_hot(frame_labels)

def retrive_batch(start_index, end_index, labels, image_sequence_folder):
    frames = []
    frame_labels = []
    for i in range(start_index, end_index):
        image_path = os.path.join(image_sequence_folder, 'video-{:09d}.png'.format(i))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
        frame_labels.append(generate_labels(labels['labels'], i))
    return start_index, frames, frame_labels

class ImageSequenceIterator():
    def __init__(self, image_sequence_folder, labels, num_processes, pool, batch_size):
        self._image_sequence_folder = image_sequence_folder
        self._labels = labels
        self._num_processes = num_processes
        self._number_of_frames = labels['num_frames']
        self._pool = pool
        self._batch_size = batch_size
        self._queue = Queue(5)

    def batchiter(self):
        batched_frames = torch.FloatTensor(self._batch_size, 3, 224, 224)
        batched_labels = np.ndarray(shape=(self._batch_size), dtype=np.int64)
        partial_retrive_batch = partial(retrive_batch, labels=self._labels, image_sequence_folder=self._image_sequence_folder)
        process_batch = int(self._batch_size / self._num_processes)
        inputs = []
        for i in range(0, self._number_of_frames, self._batch_size):
            for j in range(1, self._batch_size + 1, process_batch):
                if i + j + process_batch < self._number_of_frames:
                    inputs.append((i + j, i + j + process_batch))
                else:
                    inputs.append((i + j, self._number_of_frames))
                    break
        outputs = self._pool.starmap_async(partial_retrive_batch, inputs)
        current_index = 0
        for output in outputs.get():
            index, images, labels = output
            for image, label in zip(images, labels):
                batched_frames[current_index, :, :, :] = transform(image)
                batched_labels[current_index] = label
                current_index += 1
            if current_index == self._batch_size:
                yield batched_frames.narrow(0, 0, current_index), torch.from_numpy(batched_labels[:current_index])
                current_index = 0
        yield batched_frames.narrow(0, 0, current_index), torch.from_numpy(batched_labels[:current_index])
