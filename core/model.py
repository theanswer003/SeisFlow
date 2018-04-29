# -*- coding: utf-8 -*-

from core.FFT_MA_3D import simulate
import tensorflow as tf
from conf.settings import *


class Model(object):
    """
    Model that describles the workflow of the inverse problem
    """

    def __init__(self, wavelet, seismic, init_AI, regularization_weight, learning_rate):
        """
        initilization of the Model object
        :param wavelet: wavelet matrix (numpy array)
        :param seismic: real seismic data (numpy array)
        :param init_AI: init AI model (numpy array)
        :param regularization_weight: regularization weight in loss function
        :param learning_rate: learning rate for updating
        """

        # create the computational graph
        self._create_model(wavelet, seismic, init_AI, regularization_weight, learning_rate)

    def _create_model(self, wavelet, seismic, init_AI, regularization_weight, learning_rate):
        # simulation parameters
        lhx = SIMULATION_PARAS['lhx']
        lhy = SIMULATION_PARAS['lhy']
        lhv = SIMULATION_PARAS['lhv']

        # shape of the model
        nsample, nline, ncdp = seismic.shape

        self.wavelet = tf.constant(wavelet, dtype=tf.float32, name='wavelet')
        self.seismic = tf.constant(seismic.reshape(-1, nline * ncdp), dtype=tf.float32, name='seismic')
        self.init_AI = tf.constant(init_AI.reshape(-1, nline * ncdp), dtype=tf.float32, name='init_AI')

        # simulate the prior reservoir model
        with tf.name_scope('Simulation'):
            self.sim_AI = tf.Variable(simulate(init_AI, lhx, lhy, lhv).reshape(-1, nline * ncdp),
                                      dtype=tf.float32, name='simAI')

        # predict seismic response from the prior model
        with tf.name_scope('Synthetic'):
            self.refl = (self.sim_AI[1:] - self.sim_AI[:-1]) / (self.sim_AI[:-1] + self.sim_AI[1:])
            self.syn_seis = tf.matmul(self.wavelet, self.refl, name='syn_seis')

        # define the loss function
        with tf.name_scope('Loss'):
            data_misfit = tf.reduce_mean(tf.square(self.syn_seis - self.seismic[1:, :]), axis=0, name='data_misfit')
            model_misfit = tf.reduce_mean(tf.square(self.sim_AI - self.init_AI), axis=0, name='model_misfit')
            self.loss = data_misfit  + regularization_weight * model_misfit/100

        # define the optimizer
        with tf.name_scope('Optimization'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)