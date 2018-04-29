# -*- coding: utf-8 -*-

import numpy as np

def construct_corr_func(lh1, lh2, lv, size):
    """
    construct correlation function
    :param lh1: correlation range in X direction
    :param lh2: correlation range in Y direction
    :param lv: correlation range in Z direction
    :param size: grid size
    :return: correlation matrix
    """
    ordem = 3
    desvio = 1.0 / 4
    I, J, K = size
    X, Y, Z = np.mgrid[0:I, 0:J, 0:K]
    orig_X, orig_Y, orig_Z = np.round(I / 2), np.round(J / 2), np.round(K / 2)
    r = np.sqrt(((X - orig_X) / (3 * lh1)) ** 2 + ((Y - orig_Y) / (3 * lh2)) ** 2 + ((Z - orig_Z) / (3 * lv)) ** 2)
    value = np.zeros((I, J, K))
    value[r < 1] = 1 - 1.5 * r[r < 1] + 0.5 * r[r < 1] ** 3
    value_window = np.exp(-(np.abs((X - np.round(I / 2)) / (desvio * I)) ** ordem
                                        + np.abs((Y - np.round(J / 2)) / (desvio * J)) ** ordem
                                        + np.abs((Z - np.round(K / 2)) / (desvio * K)) ** ordem))
    corr_func = value * value_window
    return corr_func


def FFT_MA_3D(corr_func, noise):
    c2 = np.fft.ifftn(np.sqrt(np.abs(np.fft.fftn(corr_func))) * np.fft.fftn(noise))
    simulation = np.real(c2)
    return simulation


def simulate(Ip_mod, lh1, lh2, lv, nsig=1):
    std = np.std(Ip_mod)
    corr_func = construct_corr_func(lh1, lh2, lv, size=Ip_mod.shape)
    noise = np.random.randn(*Ip_mod.shape)
    simulation = Ip_mod + nsig * std * FFT_MA_3D(corr_func, noise)
    return simulation