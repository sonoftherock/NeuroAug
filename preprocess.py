import numpy as np
import tensorflow as tf

def ignore_negative(data_dir):
    dataset = np.load('./data/' + data_dir + '/original.npy')
    dataset = np.clip(dataset, 0, 1)

    # Make sure every node is connected to itself.
    for subject in dataset:
        np.fill_diagonal(subject, 1)
    np.save('./data/' + data_dir + '/ignore_negative.npy', dataset)

def add_min(data_dir):
    dataset = np.load('./data/' + data_dir + '/original.npy')
    threshold = np.amin(dataset)
    dataset = dataset + np.abs(threshold)

    # Make sure every node is connected to itself.
    for subject in dataset:
        np.fill_diagonal(subject, 1 + np.abs(threshold))
    np.save('./data/' + data_dir + '/add_min_adj.npy', dataset)
