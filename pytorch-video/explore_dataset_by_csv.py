import os
import cv2
import argparse
import pandas as pd

def get_video_fps_and_duration(filename):
    video = cv2.VideoCapture(filename)

    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return duration, frame_count


def read_videos_and_labels_from_csv(data_path, videos_prefix):
    dataset = []
    loaded_data = pd.read_csv(data_path, dtype={"id": int, "file_path": str, "start_frame": int, "end_frame": int, "label_id": int, "label_name": str})
    for _, values in loaded_data.groupby('id'):
        path = os.path.join(videos_prefix, values['file_path'].unique()[0])
        labels = []
        for _, row in values.iterrows():
            labels.append({'start_frame': row['start_frame'], 'end_frame': row['end_frame'], 'label_id': row['label_id']})
        dataset.append((path, labels))
    return dataset


parser = argparse.ArgumentParser(description='train a lstm over a videos dataset using grid search')
parser.add_argument('--dataset', help='the dataset csv file')
parser.add_argument('--videos_prefix', help='the dataset csv file')

args = parser.parse_args()
dataset_path = args.dataset
videos_prefix = args.videos_prefix

dataset = read_videos_and_labels_from_csv(dataset_path, videos_prefix)
classes_distribution = [0 for _ in range(10)]

for video_path, labels in dataset:
    _, frame_count = get_video_fps_and_duration(video_path)
    anomlies_in_video = 0
    for label in labels:
        classes_distribution[label['label_id']] += label['end_frame'] - label['start_frame']
        anomlies_in_video += label['end_frame'] - label['start_frame']
    classes_distribution[0] += frame_count - anomlies_in_video

print(classes_distribution)


