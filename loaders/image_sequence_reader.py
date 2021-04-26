import torch
from torchvision import transforms
from moviepy.editor import VideoFileClip
import os
import numpy as np
from random import shuffle
import cv2
import concurrent.futures
import logging

transform = transforms.Compose([transforms.ToTensor(),
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
        batched_frames = torch.cuda.FloatTensor(self._batch_size, 3, 224, 224)
        batched_labels = torch.cuda.LongTensor(self._batch_size)
        shuffle(self._dataset)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_processes) as executor:
            for video_num, labels in self._dataset:
                image_sequence_folder = os.path.join(self._video_folder, 'video_{}'.format(video_num))
                logging.debug('starting video {}, num frames : {}'.format(video_num, labels['num_frames']))
                yield ImageSequenceIterator(image_sequence_folder, labels, self._num_processes, executor, self._batch_size, batched_frames, batched_labels)

def generate_labels(labels_in_frames, video_frame_index):
    for label in labels_in_frames:
        if int(label[1]) <= video_frame_index <= int(label[2]):
            return 1
    return 0

def retrive_batch(start_index, end_index, labels, image_sequence_folder):
    frames = []
    frame_labels = []
    for i in range(start_index, end_index):
        image_path = os.path.join(image_sequence_folder, 'video-{:09d}.png'.format(i))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(transform(cv2.resize(image, (224, 224)) / 255))
        frame_labels.append(generate_labels(labels['labels'], i))
    return frames, frame_labels

class ImageSequenceIterator():
    def __init__(self, image_sequence_folder, labels, num_processes, pool, batch_size, batched_frames, batched_labels):
        self._image_sequence_folder = image_sequence_folder
        self._labels = labels
        self._num_processes = num_processes
        self._number_of_frames = labels['num_frames']
        self._pool = pool
        self._batch_size = batch_size
        self._batched_frames = batched_frames
        self._batched_labels = batched_labels

    def batchiter(self):
        process_batch = int(self._batch_size / self._num_processes)
        current_index = 0
        inputs = []
        for i in range(0, self._number_of_frames, self._batch_size):
            for j in range(1, self._batch_size + 1, process_batch):
                if i + j + process_batch < self._number_of_frames:
                    inputs.append((i + j, i + j + process_batch))
                else:
                    inputs.append((i + j, self._number_of_frames))
                    break
        results = self._pool.map(lambda x : retrive_batch(x[0], x[1], self._labels, self._image_sequence_folder), inputs)
        for output in results:
            images, labels = output
            for image, label in zip(images, labels):
                self._batched_frames[current_index, :, :, :] = image
                self._batched_labels[current_index] = label
                current_index += 1
            if current_index == self._batch_size:
                yield self._batched_frames, self._batched_labels
                current_index = 0
        if current_index != 0:
            yield self._batched_frames.narrow(0, 0, current_index), self._batched_labels.narrow(0, 0, current_index)
