from collections import defaultdict
import os
import pickle

labels_folder = 'D:/Dataset/Preprocessed-Labels'

#load the preprocessed dataset
with open(os.path.join(labels_folder, 'label_without_normal_videos.pkl'), 'rb') as f:
    loaded_dataset = pickle.load(f)

print(loaded_dataset)

counter_dict = defaultdict(int)
for video, attr in loaded_dataset.items():
    if not attr['labels']:
        counter_dict['normal'] += 1
    else:
        for label in attr['labels']:
            counter_dict[label[0]] += 1

print(counter_dict)

video_times_per_category = defaultdict(int)
for video, attr in loaded_dataset.items():
    if not attr['labels']:
        video_times_per_category['normal'] += attr['num_frames']
    else:
        total_anomaly_duration = 0
        for label in attr['labels']:
            start_frame = int(label[1])
            end_frame = int(label[2])
            duration = (label[2] - label[1])
            video_times_per_category[label[0]] += duration
            total_anomaly_duration += duration
        video_times_per_category['normal'] += attr['num_frames'] - total_anomaly_duration


print(video_times_per_category)
print(sum(map(lambda x: x[1] if x[0] != 'normal' else 0, video_times_per_category.items())))
                