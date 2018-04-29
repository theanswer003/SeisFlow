# -*- coding: utf-8 -*-

from core.model import Model
import tensorflow as tf
from conf.settings import *
from core.FFT_MA_3D import *


ngpu = DEVICE_PARAS['gpu_num']           # number of gpu available, if 0, CPU will be used for computation.
nsim = SIMULATION_PARAS['sim_num']       # number of realizations
graph_path = OUTPUT['graphs']            # path of data flow graph

nsim_per_gpu = nsim // ngpu

lhx = SIMULATION_PARAS['lhx']            # correlation range in X direction
lhy = SIMULATION_PARAS['lhy']            # correlation range in Y direction
lhv = SIMULATION_PARAS['lhv']            # correlation range in Z direction

learning_rate = HYPER_PARAS['learning_rate']
training_steps = HYPER_PARAS['training_steps']   # updating steps
regularization_weight = HYPER_PARAS['regularization_weight']

# define the computing devices
if 0 == ngpu:
    devices = ['cpu:0']
else:
    devices = ['gpu:%d'%i for i in range(ngpu)]

def run():
    wavelet = np.load(DATAPATH['wavelet'])
    seismic = np.load(DATAPATH['seismic'])     # nsample, nline, ncdp
    init_AI = np.load(DATAPATH['init_AI'])     # nsample, nline, ncdp

    for epoch in range(nsim_per_gpu):
        tf.reset_default_graph()      # clear the tensorflow graph

        models = []

        for i in range(ngpu):
            with tf.device(devices[i]):  # allocate computing devices for different models
                with tf.name_scope('Realization_%d' % (i + epoch * ngpu)):
                    model = Model(wavelet, seismic, init_AI,
                                  regularization_weight=regularization_weight,
                                  learning_rate=learning_rate)
                    models.append(model)

        # create a session to run the data flow graph
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            training_steps = HYPER_PARAS['training_steps']
            for step in range(training_steps):
                sess.run([model.train_step for model in models])

            realizations = sess.run([model.sim_AI for model in models])

            # save the updated realizations
            for i in range(len(realizations)):
                save_path = os.path.join(OUTPUT['results'], 'realization_%d.npy' % (epoch * ngpu + i))
                np.save(save_path, realizations[i])