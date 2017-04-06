#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def load_train_data():
    data = np.loadtxt(open('data/train.csv'), dtype=np.float, delimiter=',', skiprows=1)
    label = data[:, 0]
    data = data[:, 1:]
    return data, label


def load_test_data():
    data = np.loadtxt(open('data/test.csv'), dtype=np.float, delimiter=',', skiprows=1)
    return data


def save_results(result):
    idx = np.array([range(1, result.shape[0]+1)]).transpose()
    result = np.array([result]).transpose()
    np.savetxt('data/result.csv', np.hstack((idx, result)), fmt='%d', delimiter=',',
               header='ImageId,Label', comments='')