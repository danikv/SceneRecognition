import argparse
from collections import defaultdict
import utils
import os
import json
import pandas as pd
from copy import deepcopy

parser = argparse.ArgumentParser(description='convert video time labels into frame labales')
parser.add_argument('--labels_folder', help='labels folder')
parser.add_argument('--videos_folder', help='videos folder')
parser.add_argument('--fps', type=int, help='frames per second when converting video to images')
parser.add_argument('--output_file', help='features output file')

LABEL_MAP = {
    "none": 0,
    "fighting": 1,
    "robbery": 2,
    "stealing": 3,
    "shoplifting": 4,
    "burglary": 5,
    "arsen": 6,
    "pouring gas": 6,
    "explosion": 7,
    "vandalism": 8,
    "arrest": 9,
    "gun pointing": 10,
    "shooting": 10
}

args = parser.parse_args()
labels_folder = args.labels_folder
videos_folder = args.videos_folder
fps = args.fps

dataset = []
for (dirpath, dirnames, filenames) in os.walk(labels_folder):
    for file in filenames:
        with open(os.path.join(labels_folder, file)) as f:
            current_video = {}
            number = utils.get_number_from_filename(file)
            labels = json.load(f)['labels']
            current_video['id'] = number
            current_video['file_path'] = f'video_{number}.mp4'
            labels_as_frames = []
            for label in labels:
                start_time_min, start_time_sec = utils.parse_time(label[1])
                end_time_min, end_time_sec = utils.parse_time(label[2])
                start_frame = (start_time_min * 60 + start_time_sec) * fps
                end_frame = (end_time_min * 60 + end_time_sec) * fps
                current_video['start_frame'] = start_frame
                current_video['end_frame'] = end_frame
                current_video['label_id'] = LABEL_MAP[label[0]]
                current_video['label_name'] = label[0]
                dataset.append(deepcopy(current_video))

dataframe = pd.DataFrame(dataset)
dataframe.to_csv(args.output_file)



