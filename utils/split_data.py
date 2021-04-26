from sklearn.model_selection import train_test_split
import pickle
import os
import argparse


parser = argparse.ArgumentParser(description='train a lstm over a videos dataset')
parser.add_argument('--labels_file', help='the unified labels file')
parser.add_argument('--output_folder', help='output folder for the train and test split files')

args = parser.parse_args()

#load the preprocessed dataset
with open(args.labels_file, 'rb') as f:
    loaded_dataset = pickle.load(f)


#filter irellevent categories
for video, atrr in loaded_dataset.items():
    atrr['labels'] = [label for label in atrr['labels'] if label[0] != 'pouring gas' and label[0] != 'gun pointing']
print(len(loaded_dataset))
loaded_dataset = dict(filter(lambda x : len(x[1]['labels']) != 0, loaded_dataset.items()))
print(len(loaded_dataset))
#separate data to test and train
train_indexes, test_indexes = train_test_split(range(len(loaded_dataset)), test_size=0.2)
train = [(key, value) for key,value in loaded_dataset.items() if key in train_indexes]
test = [(key, value) for key,value in loaded_dataset.items() if key in test_indexes]

for key, value in train:
    if not value['labels']:
        value['num_frames'] = int(value['num_frames'] / 8)

with open(os.path.join(args.output_folder, 'train_without_normal.pkl'), 'wb') as f:
    pickle.dump(train, f)

with open(os.path.join(args.output_folder, 'test_without_normal.pkl'), 'wb') as f:
    pickle.dump(test, f)

#check if data is balanced
normal_video_frames = 0
anomaly_video_frames = 0
for key, value in train:
    anomaly = 0
    for label in value['labels']:
        anomaly += label[2] - label[1]
    normal_video_frames += value['num_frames'] - anomaly
    anomaly_video_frames += anomaly

print(normal_video_frames)
print(anomaly_video_frames)
print(anomaly_video_frames / (anomaly_video_frames + normal_video_frames))

