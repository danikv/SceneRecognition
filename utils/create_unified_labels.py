import argparse
import pickle
import utils
from collections import defaultdict
import os
import json

parser = argparse.ArgumentParser(description='convert video time labels into frame labales')
parser.add_argument('--labels_folder', help='labels folder')
parser.add_argument('--video_images_folder', help='videos as images folder')
parser.add_argument('--fps', type=int, help='frames per second when converting video to images')
parser.add_argument('--output_file', help='features output file')

args = parser.parse_args()
labels_folder = args.labels_folder
video_images_folder = args.video_images_folder
fps = args.fps

dataset = defaultdict(dict)
for (dirpath, dirnames, filenames) in os.walk(labels_folder):
    for file in filenames:
        with open(os.path.join(labels_folder, file)) as f:
            number = utils.get_number_from_filename(file)
            labels = json.load(f)['labels']
            if len(labels) == 0:
                continue
            current_images_folder = os.path.join(video_images_folder, 'video_{}'.format(number))
            dataset[number]['num_frames'] = len(os.listdir(current_images_folder))
            labels_as_frames = []
            for label in labels:
                start_time_min, start_time_sec = utils.parse_time(label[1])
                end_time_min, end_time_sec = utils.parse_time(label[2])
                start_frame = (start_time_min * 60 + start_time_sec) * fps
                end_frame = (end_time_min * 60 + end_time_sec) * fps
                labels_as_frames.append((label[0], int(start_frame), int(end_frame)))
            dataset[number]['labels'] = labels_as_frames

dataset = dict(dataset)
with open(args.output_file, 'wb') as f:
    pickle.dump(dataset, f)



