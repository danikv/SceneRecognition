import numpy as np
import os
import json
import pickle
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
import cv2
from multiprocessing import Process, Queue
from model import BaseModel
import torch
import torch.optim as optim
import torch.nn as nn
from image_sequence_reader import ImageSequenceDataset
import argparse
from sklearn.model_selection import train_test_split
import logging

classes = ['normal', 'anomaly']


def train_model(model, training_generator, device, optimizer, criterion, batch_size, epoch):
    current_loss = 0.0
    num_frames = 0.0
    for i, data in enumerate(training_generator, 0):
        # get the inputs; data is a list of [inputs, labels]
        model.init_hidden(device)
        running_loss = 0.0
        for inputs, labels in data.batchiter(batch_size):
            inputs, labels = inputs.to(device), labels.to(device)

            model._hidden[0].detach_()
            model._hidden[1].detach_()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            num_frames += batch_size
            current_loss += loss.item()
            running_loss += loss.item()
        logging.info('[%d, %5d] loss: %.3f' %
                (epoch + 1, i, running_loss))
    return current_loss / num_frames

def evaluate_model(model, evaluation_generator, device, batch_size):
    num_labels = 0.0
    correct_labels = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(evaluation_generator, 0):
            model.init_hidden(device)
            for inputs, labels in data.batchiter(batch_size):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                for i, label in enumerate(labels):
                    if label == outputs[i]:
                        correct_labels += 1
                    num_labels += 1
    return correct_labels / num_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a lstm over a videos dataset')
    parser.add_argument('--ephocs', type=int, help='the number of ephocs')
    parser.add_argument('--dataset', help='the dataset folder which contains 2 folders, videos and labels')
    parser.add_argument('--output_file', help='output file to write loss and evaluation accuracy')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--num_processes', type=int, help='number of processes to use during trainging')
    parser.add_argument('--logger_file', help='output file for loggings')

    args = parser.parse_args()


    dataset_folder = args.dataset
    labels_folder = os.path.join(dataset_folder, 'labels')
    videos_folder = os.path.join(dataset_folder, 'videos-Images')
    ephocs = args.ephocs
    batch_size = args.batch_size
    num_processes = args.num_processes
    logging_output_file = args.logger_file

    logging.basicConfig(filename=logging_output_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    train = []
    with open(os.path.join(labels_folder, 'train.pkl'), 'rb') as f:
        train = pickle.load(f)

    test = []
    with open(os.path.join(labels_folder, 'test.pkl'), 'rb') as f:
        test = pickle.load(f)

    train, evaluation = train_test_split(train, test_size=0.2)

    dataset_train = ImageSequenceDataset(videos_folder, train, num_processes)
    #training_generator = torch.utils.data.DataLoader(dataset_train)


    dataset_evaluation = ImageSequenceDataset(videos_folder, evaluation, num_processes)
    #evaluation_generator = torch.utils.data.DataLoader(dataset_evaluation)

    dataset_test = ImageSequenceDataset(videos_folder, test, num_processes)
    #test_generator = torch.utils.data.DataLoader(dataset_test)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    net = BaseModel(512, 3)
    net.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train(True)

    #start training
    torch.autograd.set_detect_anomaly(True)
    evaluation_accuracy = []
    loss = []

    logging.info('start training')
    for epoch in range(ephocs):  # loop over the dataset multiple times
        logging.info('starting epoch {}'.format(epoch))
        loss.append(train_model(net, dataset_train, device, optimizer, criterion, batch_size, epoch))
        #calculate accuracy over the evaluation dataset
        logging.info('starting evaluation {}'.format(epoch))
        evaluation_accuracy.append(evaluate_model(net, dataset_evaluation, device, batch_size))            
    test_accuracy = evaluate_model(net, dataset_test, device, batch_size)

    file_data = {}
    file_data['test_accuracy'] = test_accuracy
    file_data['eval_accuracy'] = evaluation_accuracy
    file_data['train_loss'] = loss


    with open(args.output_file, 'wb') as f:
        pickle.dump(file_data, f)

    print('Finished Training')