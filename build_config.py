# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 16:23:01 2021

@author: femiogundare
"""

import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='path to create config file')
args = parser.parse_args()


params = {
    'conv_subsample_lengths' : [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    'conv_filter_length' : 16,
    'conv_num_filters_start' : 32,
    'conv_init' : "he_normal",
    'conv_activation' : "relu",
    'conv_dropout' : 0.2,
    'conv_num_skip' : 2,
    'conv_increase_channels_at' : 4,

    'learning_rate' : 0.001,
    'batch_size' : 2,
    
    
    'train' : 'C:\\Users\\Dell\\Desktop\\CV Projects\\ecg\\data/train.json', 
    'val' : 'C:\\Users\\Dell\\Desktop\\CV Projects\\ecg\\data/val.json', 
    'test' : 'C:\\Users\\Dell\\Desktop\\CV Projects\\ecg\\data/test.json',

    'generator' : True,

    'save_dir' : 'C:\\Users\\Dell\\Desktop\\CV Projects\\ecg/saved'
}



f = open(args.config_file, 'w')
f.write(json.dumps(params))
f.close()