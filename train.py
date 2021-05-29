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
from loaders.image_sequence_reader import ImageSequenceDataset
import argparse
from sklearn.model_selection import train_test_split
import logging
from importlib import import_module
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

classes = ['normal', 'anomaly']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a lstm over a videos dataset')
    parser.add_argument('--epochs', type=int, help='the number of epochs')
    parser.add_argument('--dataset', help='the dataset folder which contains 2 folders, videos and labels')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--num_processes', type=int, help='number of processes to use during trainging')
    parser.add_argument('--logger_file', help='output file for loggings')
    parser.add_argument('--model_save_path', help='path for model saving')
    parser.add_argument('--stats_file', help='path for tensor board')
    parser.add_argument('--model_class', help='model class')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--hidden_dim', type=int, help='hidden dim')
    parser.add_argument('--clip', type=float, help='gradient clipping value')


    args = parser.parse_args()


    dataset_folder = args.dataset
    labels_folder = os.path.join(dataset_folder, 'Preprocessed-Labels')
    videos_folder = os.path.join(dataset_folder, 'Videos-Images-1fps')
    epochs = args.epochs
    batch_size = args.batch_size
    num_processes = args.num_processes
    logging_output_file = args.logger_file
    model_prefix = args.model_prefix
    hidden_dim = args.hidden_dim
    clip = args.clip
    stats_file = args.stats_file
    model_class = getattr(import_module('models.{}'.format(args.model_class.lower())), args.model_class)
    lr = args.lr
    model_path = os.path.join('{}-{}-{}-{}', args.model_save_path, lr, hidden_dim, clip)

    logging.basicConfig(filename=logging_output_file, filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

    train = []
    with open(os.path.join(labels_folder, '1fps_100frames_train.pkl'), 'rb') as f:
        train = pickle.load(f)

    test = []
    with open(os.path.join(labels_folder, '1fps_100frames_test.pkl'), 'rb') as f:
        test = pickle.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train, evaluation = train_test_split(train, test_size=0.2)

    dataset_train = ImageSequenceDataset(videos_folder, train, num_processes, batch_size)
    #training_generator = torch.utils.data.DataLoader(dataset_train)


    dataset_evaluation = ImageSequenceDataset(videos_folder, evaluation, num_processes, batch_size)
    #evaluation_generator = torch.utils.data.DataLoader(dataset_evaluation)

    dataset_test = ImageSequenceDataset(videos_folder, test, num_processes, batch_size)
    #test_generator = torch.utils.data.DataLoader(dataset_test)


    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    model = model_class(2)
    model.to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
    model.train(True)


    logging.info('start training')
    best_accuracy = 0
    best_model_path = None
    comment = f'gradient_clip = {clip} lr = {lr} hidden_dim = {hidden_dim} ephochs = {epochs}'
    tb = SummaryWriter(log_dir=os.path.join(stats_file, comment))
    for epoch in range(epochs):  # loop over the dataset multiple times
        logging.info('starting epoch {}'.format(epoch))
        train_metrics = model.train_model(dataset_train, device, optimizer, criterion, epoch))
        #calculate accuracy over the evaluation dataset
        val_metrics = model.evaluate_model(dataset_evaluation, device, criterion)) 
        if val_metrics[-1][1] > best_accuracy:
           best_accuracy = val_metrics[-1][1]
           best_model_path = os.path.join(model_path, '{}-{}-{}.model'.format(model_prefix, best_accuracy, lr))
           torch.save(model.state_dict(), best_model_path)
        tb.add_scalar("Train Loss", train_metrics[-1][0], epoch)
        tb.add_scalar("Train Accuracy", train_metrics[-1][1], epoch)
        tb.add_scalar("Validation Loss", val_metrics[-1][0], epoch)
        tb.add_scalar("Validation Accuracy", val_metrics[-1][1], epoch)
    model.load_state_dict(torch.load(best_model_path))
    test_data = model.evaluate_model(dataset_test, device, criterion)

    print('Finished Training')
