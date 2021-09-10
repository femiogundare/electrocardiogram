# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 19:16:40 2021

@author: femiogundare
"""

import os
import random
import json
import numpy as np
from scipy import io as sio
import tqdm

STEP = 256

def load_ecg_mat(ecg_file):
    """
    Loads ECG matlab file
    """
    return sio.loadmat(ecg_file)['val'].squeeze()


def load_all(data_path):
    """
    Loads all data
    """
    label_file = os.path.join(data_path, 'REFERENCE-v3.csv')
    with open(label_file, 'r') as file:
        records = [l.strip().split(',') for l in file]
        
    dataset = []
    for record, label in tqdm.tqdm(records):
        ecg_file = os.path.join(data_path, record + ".mat")
        ecg_file = os.path.abspath(ecg_file)
        ecg = load_ecg_mat(ecg_file)
        num_labels = ecg.shape[0] // STEP
        dataset.append((ecg_file, [label]*num_labels))
    return dataset 



def split_data(dataset, val_frac, test_frac):
    val_cut = int(val_frac * len(dataset))
    test_cut = int(test_frac * len(dataset))
    random.shuffle(dataset)
    val = dataset[:val_cut]
    test = dataset[-test_cut:]
    train = dataset[val_cut : -test_cut]
    print(len(train), len(val), len(test))
    return train, val, test



def make_json(save_path, dataset):
    with open(save_path, 'w') as file:
        for entry in dataset:
            datum = {'ecg' : entry[0], 'labels' : entry[1]}
            json.dump(datum, file)
            file.write('\n')
            

if __name__ == '__main__':
    random.seed(42)
    
    val_frac = 0.1
    test_frac = 0.1
    data_path = 'C:\\Users\\Dell\\Desktop\\CV Projects\\ecg\\data'
    dataset = load_all(data_path+'/training2017')
    train, val, test = split_data(dataset, val_frac, test_frac)
    make_json(data_path+'/train.json', train)
    make_json(data_path+'/val.json', val)
    make_json(data_path+'/test.json', test)