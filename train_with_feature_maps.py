from models.lstm_resnet_model import LSTM_Resnet_Model
from random import shuffle
import numpy as np
import os
import json
import pickle
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
from loaders.image_sequence_reader import ImageSequenceDataset, generate_feature_maps_dataset, generate_labels
import argparse
from sklearn.model_selection import train_test_split
import logging
from importlib import import_module
from matplotlib import pyplot as plt
import itertools
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

classes = ['normal', 'anomaly']



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a lstm over a videos dataset using grid search')
    parser.add_argument('--epochs', type=int, help='the number of epochs')
    parser.add_argument('--dataset', help='the dataset folder which contains 2 folders, videos and labels')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--num_processes', type=int, help='number of processes to use during trainging')
    parser.add_argument('--logger_file', help='output file for loggings')
    parser.add_argument('--model_save_path', help='path for model saving')
    parser.add_argument('--stats_file', help='path for tensor board')
    parser.add_argument('--frames', type=int, help='number of frames in video')
    parser.add_argument('--fps', type=int, help='fps for video')

    args = parser.parse_args()


    dataset_folder = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    num_processes = args.num_processes
    logging_output_file = args.logger_file
    stats_file = args.stats_file
    prefix_path = args.model_save_path
    frames = args.frames
    fps = args.fps
    lr_list = [0.0001, 0.00005]
    clip_list = [0.5, 0.1]
    hidden_dim_list = [2048, 4096]
    labels_folder = os.path.join(dataset_folder, 'Preprocessed-Labels')
    videos_folder = os.path.join(dataset_folder, f'Videos-Images-{fps}fps-FeatureMaps')

    logging.basicConfig(filename=logging_output_file, filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

    train = []
    with open(os.path.join(labels_folder, '{}fps_{}frames_train.pkl'.format(fps, frames)), 'rb') as f:
        train = pickle.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=4)
    train_dataset = generate_feature_maps_dataset(train, videos_folder)


    logging.info('start training')
    best_model_path = None
    for lr, hidden_dim, clip in itertools.product(lr_list, hidden_dim_list, clip_list):
        for index , (train_index, val_index) in enumerate(kf.split(train_dataset)):
            current_train = [train_dataset[i] for i in train_index]
            current_val = [current_val[i] for i in val_index]
            current_train = DataLoader(current_train, shuffle=True, batch_size=1)
            current_val = DataLoader(current_val, shuffle=True, batch_size=1)

            comment = f'gradient_clip = {clip} lr = {lr} hidden_dim = {hidden_dim} ephochs = {epochs} cross_val = {index} frames = {frames} fps = {fps}'
            tb = SummaryWriter(log_dir=os.path.join(stats_file, comment))
            model = LSTM_Resnet_Model(2, hidden_dim=hidden_dim)
            model.to(device)

            criterion = nn.CrossEntropyLoss().cuda()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model.train(True)
            best_accuracy = 0
            logging.info('stating with hypter params , lr {} , hidden_dim {} , clip {}, cross_val = {}'.format(lr, hidden_dim, clip, index))
            model_path = '{}-{}-{}-{}.model'.format(prefix_path, lr, hidden_dim, clip)
            for epoch in range(epochs):  # loop over the dataset multiple times
                logging.info('starting epoch {}'.format(epoch))
                train_metrics = model.train_model_feature_maps(current_train, device, optimizer, criterion, clip)
                #calculate accuracy over the evaluation dataset
                val_metrics = model.evaluate_model_feature_maps(current_val, device, criterion)
                if val_metrics[1] > best_accuracy:
                    best_accuracy = val_metrics[1]
                    torch.save(model.state_dict(), model_path)
                tb.add_scalar("Train Loss", train_metrics[0], epoch)
                tb.add_scalar("Train Accuracy", train_metrics[1], epoch)
                tb.add_scalar("Validation Loss", val_metrics[0], epoch)
                tb.add_scalar("Validation Accuracy", val_metrics[1], epoch)

    print('Finished Training')
