from collections import defaultdict


#load the preprocessed dataset
with open(os.path.join(labels_folder) + 'unifed_labels2.pkl', 'rb') as f:
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
        video_times_per_category['normal'] += attr['duration']
    else:
        for label in attr['labels_in_frames']:
            fps = attr['fps']
            start_frame = int(label[1])
            end_frame = int(label[2])
            duration = (label[2] - label[1]) / fps
            video_times_per_category[label[0]] += duration
                