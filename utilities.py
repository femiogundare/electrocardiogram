# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 20:48:24 2021

@author: femiogundare
"""

import os
import pickle


"""
def load(dirname):
    preprocessor_file = os.path.join(dirname, "preprocessor.bin")
    with open(preprocessor_file, 'r', encoding='Latin-1') as file:
        preprocessor = pickle.load(file)
    return preprocessor
"""


def load(dirname):
    preprocessor_file = os.path.join(dirname, "preprocessor.bin")
    preprocessor = pickle.loads(open(preprocessor_file, 'rb').read())
    return preprocessor

"""
def save(preprocessor, dirname):
    preprocessor_file = os.path.join(dirname, 'preprocessor.bin')
    with open(preprocessor_file, 'w') as file:
        pickle.dump(preprocessor, file)
"""
     
        
def save(preprocessor, dirname):
    preprocessor_file = os.path.join(dirname, 'preprocessor.bin')
    f = open(preprocessor_file, 'wb')
    f.write(pickle.dumps(preprocessor))
    f.close()