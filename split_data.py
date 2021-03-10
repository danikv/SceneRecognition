from sklearn.model_selection import train_test_split
import pickle
import os
import argparse


parser = argparse.ArgumentParser(description='train a lstm over a videos dataset')
parser.add_argument('--dataset', help='the dataset folder which contains 2 folders, videos and labels')

args = parser.parse_args()

labels_folder = os.path.join(dataset_folder, 'labels')

#load the preprocessed dataset
with open(os.path.join(labels_folder, 'unifed_labels.pkl'), 'rb') as f:
    loaded_dataset = pickle.load(f)

print(loaded_dataset)


#filter irellevent categories
for video, atrr in loaded_dataset.items():
    atrr['labels'] = [label for label in atrr['labels'] if label[0] != 'pouring gas' and label[0] != 'gun pointing']
    atrr['labels_in_frames'] = [label for label in atrr['labels_in_frames'] if label[0] != 'pouring gas' and label[0] != 'gun pointing']


#separate data to test and train
train_indexes, test_indexes = train_test_split(range(len(loaded_dataset)), test_size=0.2)
train = [(key, value['labels_in_frames']) for key,value in loaded_dataset.items() if key in train_indexes]
test = [(key, value['labels_in_frames']) for key,value in loaded_dataset.items() if key in test_indexes]