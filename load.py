# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 21:04:34 2021

@author: femiogundare
"""

import os
import random
import json
import argparse
import numpy as np
import tqdm
from scipy import io as sio
from tensorflow import keras


STEP = 256

def data_generator(batch_size, preprocessor, x, y):
    """Generates data in batches"""
    num_examples = len(x)
    examples = zip(x, y)
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    random.shuffle(batches)
    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preprocessor.process(x, y)
            
            
def load_ecg(record):
    """Loads an ECG record"""
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        # If record is in matlab format
        ecg = sio.loadmat(record)['val'].squeeze()
    else: 
        # Assumes binary 16 bit integers
        with open(record, 'r') as file:
            ecg = np.fromfile(file, dtype=np.int16)
    
    # Truncate the ecg record to handle examples that are not a multiple of STEP
    trunc_samp = STEP * int(len(ecg) // STEP)
    return ecg[:trunc_samp]
            
     
def load_dataset(data_json):
    """Loads an ECG file containing ECG records and their corresponding labels"""
    with open(data_json, 'r') as file:
        data = [json.loads(l) for l in file]
    
    ecgs = []; labels = []
    
    for entry in tqdm.tqdm(data):
        ecgs.append(load_ecg(entry['ecg']))
        labels.append(entry['labels'])
        
    return ecgs, labels


def compute_mean_std(x):
    """Computes the mean and standard deviation of an ECG record"""
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32),
           np.std(x).astype(np.float32))


def pad(x, val=0, dtype=np.float32):
    """
    Padding function
    
    Args:
        x : batch of data
        val : fill value
    """
    max_len = max(len(i) for i in x) 
    #print(max_len)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    #print(padded.shape)
    #print(padded)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i  
    #print(padded)
    return padded
    

class Preprocessor:
    """Preprocessor"""
    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}
        
    def process(self, x, y):
        return self.process_x(x), self.process_y(y)
    
    def process_x(self, x):
        x = pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x
    
    def process_y(self, y):
        y = pad([[self.class_to_int[c] for c in s] for s in y], val=3, dtype=np.int32)
        y = keras.utils.to_categorical(y, num_classes=len(self.classes))
        #print(y)
        return y
        
        
if __name__ == '__main__':
    
    data_json = 'C:\\Users\\Dell\\Desktop\\CV Projects\\ecg\\data/train.json'
    train = load_dataset(data_json)
    preprocessor = Preprocessor(*train)
    batch_size = 32
    generator = data_generator(batch_size, preprocessor, *train)
    for x, y in generator:
        print(x.shape, y.shape)
        break
    