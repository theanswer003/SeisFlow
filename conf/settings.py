# -*- coding: utf-8 -*-

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE_PARAS = {
    'gpu_num': 0                               # number of gpu available, if 0, CPU will be used for computation.
}

DATABASE = {
    'engine': 'file_storage',                  # other engines in the future
    'path': os.path.join(BASE_DIR, 'db'),      # path of input
}

DATAPATH = {
    'seismic': os.path.join(DATABASE['path'], 'seismic.npy'),     # path of real seismic data
    'init_AI': os.path.join(DATABASE['path'], 'init_AI.npy'),     # path of init AI model
    'wavelet': os.path.join(DATABASE['path'], 'wavelet.npy'),        # path of wavelet
}

OUTPUT = {
    'results': os.path.join(BASE_DIR, 'results'),      # path of output
    'graphs': os.path.join(BASE_DIR, 'graphs'),        # path of data flow graph
}

HYPER_PARAS = {
    'learning_rate': 100,               # learning rate for updating
    'regularization_weight': 0.1,       # regularization weight in loss function
    'training_steps': 500
}

SIMULATION_PARAS = {
    'lhx': 10,                         # correlation range in X direction
    'lhy': 10,                         # correlation range in Y direction
    'lhv': 2,                          # correlation range in Z direction
    'sim_num': 100                     # number of realizations
}