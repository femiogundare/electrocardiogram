# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 21:01:16 2021

@author: Dell
"""

import argparse
import numpy as np
from tensorflow import keras
import os
import json

import load
import network
import utilities


def predict(data_json, model_path, params):
    preprocessor = utilities.load(os.path.dirname(model_path))
    dataset = load.load_dataset(data_json)
    x, y = preprocessor.process(*dataset)
    
    params.update({
        'input_shape' : [None, 1],
        'num_categories' : len(preprocessor.classes)
    })
    
    model = network.build_network(**params)
    model.load_weights(model_path)

    #model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)

    return probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file')
    parser.add_argument('data_json', help='path to data json')
    parser.add_argument('model_path', help='path to model')
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    probs = predict(args.data_json, args.model_path, params)
    #print(probs.shape)
    print(probs)