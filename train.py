# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 20:46:05 2021

@author: femiogundare
"""

import os
import random
import json
import time
from datetime import datetime
import argparse
import numpy as np
from tensorflow import keras

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import load
import utilities
import network


MAX_EPOCHS = 100

def make_save_dir(dirname, experiment_name):
    start_time = datetime.now().strftime('%m%d%H')
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_filename_for_saving_model(save_dir):
    return os.path.join(save_dir,
            '{val_loss:.3f}-{val_auc:.3f}-{epoch:03d}-{loss:.3f}-{auc:.3f}.hdf5')

def get_filename_for_saving_plot(save_dir):
    return os.path.join(save_dir,
            'train_val_curves.png')


def train(args, params):
    
    print('Loading training set...')
    train = load.load_dataset(params['train'])
    print('Loading validation set...')
    val = load.load_dataset(params['val'])
    print('Building preprocessor...')
    preprocessor = load.Preprocessor(*train)
    print('Training size: ' + str(len(train[0])) + ' examples.')
    print('Validation size: ' + str(len(val[0])) + ' examples.')
    
    save_dir = make_save_dir(params['save_dir'], args.experiment)
    
    utilities.save(preprocessor, save_dir)
    
    params.update({
        'input_shape' : [None, 1],
        'num_categories' : len(preprocessor.classes)
    })
    
    model = network.build_network(**params)
    print(model.summary())
    
    stopping = keras.callbacks.EarlyStopping(patience=8)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=params['learning_rate'] * 0.001)
    
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving_model(save_dir),
        save_best_only=False)

    batch_size = params.get('batch_size', 32)
    
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    if params.get('generator', False):
        train_gen = load.data_generator(batch_size, preprocessor, *train)
        val_gen = load.data_generator(batch_size, preprocessor, *val)
        H = model.fit_generator(
            train_gen,
            steps_per_epoch=int(len(train[0]) // batch_size),
            epochs=MAX_EPOCHS,
            validation_data=val_gen,
            validation_steps=int(len(val[0]) // batch_size),
            callbacks=[checkpointer, reduce_lr, stopping])
    else:
        train_x, train_y = preprocessor.process(*train)
        val_x, val_y = preprocessor.process(*val)
        H = model.fit(
            train_x, train_y,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,
            validation_data=(val_x, val_y),
            callbacks=[checkpointer, reduce_lr, stopping])
        
    plt.figure()
    plt.plot(np.arange(0, len(H.history['auc'])), H.history['auc'], '-o', label='Train AUC', color='#ff7f0e')
    plt.plot(np.arange(0, len(H.history['val_auc'])), H.history['val_auc'], '-o', label='Val AUC', color='#1f77b4')
    x = np.argmax( H.history['val_auc'] ); y = np.max( H.history['val_auc'] )
    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,'max auc\n%.2f'%y,size=14)
    plt.ylabel('AUC',size=14); plt.xlabel('Epoch',size=14)
    plt.legend(loc=2)
    plt2 = plt.gca().twinx()
    plt2.plot(np.arange(0, len(H.history['loss'])), H.history['loss'], '-o', label='Train Loss', color='#2ca02c')
    plt2.plot(np.arange(0, len(H.history['val_loss'])), H.history['val_loss'], '-o', label='Val Loss', color='#d62728')
    x = np.argmin( H.history['val_loss'] ); y = np.min( H.history['val_loss'] )
    ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)
    plt.ylabel('Loss',size=14)
    plt.title(f'Loss and AUC Curves',size=18)
    plt.legend(loc=3)
    plt.show()
    plt.savefig(get_filename_for_saving_plot(save_dir))
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file')
    parser.add_argument('--experiment', '-e', help='tag with experiment name',
                        default='default')
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    train(args, params)