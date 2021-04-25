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

classes = ['normal', 'anomaly']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a lstm over a videos dataset')
    parser.add_argument('--ephocs', type=int, help='the number of ephocs')
    parser.add_argument('--dataset', help='the dataset folder which contains 2 folders, videos and labels')
    parser.add_argument('--output_file', help='output file to write loss and evaluation accuracy')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--num_processes', type=int, help='number of processes to use during trainging')
    parser.add_argument('--logger_file', help='output file for loggings')
    parser.add_argument('--model_save_path', help='path for model saving')
    parser.add_argument('--model_class', help='model class')

    args = parser.parse_args()


    dataset_folder = args.dataset
    labels_folder = os.path.join(dataset_folder, 'Preprocessed-Labels')
    videos_folder = os.path.join(dataset_folder, 'Videos-Images')
    ephocs = args.ephocs
    batch_size = args.batch_size
    num_processes = args.num_processes
    logging_output_file = args.logger_file
    model_path = args.model_save_path
    model_class = getattr(import_module('models.{}'.format(args.model_class.lower())), args.model_class)

    logging.basicConfig(filename=logging_output_file, filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

    train = []
    with open(os.path.join(labels_folder, 'train.pkl'), 'rb') as f:
        train = pickle.load(f)

    test = []
    with open(os.path.join(labels_folder, 'test.pkl'), 'rb') as f:
        test = pickle.load(f)

    train, evaluation = train_test_split(train, test_size=0.2)

    dataset_train = ImageSequenceDataset(videos_folder, train, num_processes, batch_size)
    #training_generator = torch.utils.data.DataLoader(dataset_train)


    dataset_evaluation = ImageSequenceDataset(videos_folder, evaluation, num_processes, batch_size)
    #evaluation_generator = torch.utils.data.DataLoader(dataset_evaluation)

    dataset_test = ImageSequenceDataset(videos_folder, test, num_processes, batch_size)
    #test_generator = torch.utils.data.DataLoader(dataset_test)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    model = model_class(2)
    model.to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train(True)
    
    #start training
    evaluation_data = []
    train_data = []

    logging.info('start training')
    best_accuracy = 0
    for epoch in range(ephocs):  # loop over the dataset multiple times
        logging.info('starting epoch {}'.format(epoch))
        train_data.append(model.train_model(dataset_train, device, optimizer, criterion, epoch))
        #calculate accuracy over the evaluation dataset
        logging.info('starting evaluation {}'.format(epoch))
        evaluation_data.append(model.evaluate_model(dataset_evaluation, device, criterion)) 
        if evaluation_data[-1][1] > best_accuracy:
           best_accuracy = evaluation_data[-1][1]
           torch.save(model.state_dict(), os.path.join(model_path, 'restnet_101_{}.model'.format(best_accuracy)))
        logging.info(evaluation_data)
        logging.info(train_data)
    test_data = evaluate_model(model, dataset_test, device, criterion)

    file_data = {}
    file_data['test__data'] = test_data
    file_data['val_data'] = evaluation_data
    file_data['train_data'] = train_data

    logging.info(file_data)

    with open(args.output_file, 'wb') as f:
        pickle.dump(file_data, f)

    print('Finished Training')
