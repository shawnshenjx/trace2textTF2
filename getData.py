import os
import pandas as pd
from collections import defaultdict
import numpy as np
import glob
import tensorflow as tf
from scipy.interpolate import interp1d
from functools import partial
from preprocess import interpolate, add_noise

NN = '_'



def getDataTrain1(max_length, label_pad,  data_split):
    np_load_old = partial(np.load)

    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    input_data = np.load('data/data_fake.npy')
    label_data = np.load('data/label_fake.npy')

    tsteps_asii_ = 20

    padded_input_data = []

    for i, v in enumerate(input_data):
        v = interpolate(v, len(label_data[i]), tsteps_asii=tsteps_asii_)[:int(max_length)]
        residual = int(max_length) - v.shape[0]
        padding_array = np.zeros([int(residual), 2])
        padded_input_data.append(
            np.concatenate([v, padding_array], axis=0))

    padded_label_data = []

    for _, v in enumerate(label_data):
        v = np.array(v)[:int(label_pad)]
        residual = int(label_pad) - v.shape[0]
        padding_array = np.zeros([int(residual)])
        padded_label_data.append(
            np.concatenate([v, padding_array], axis=0))



    padded_input_data = np.stack(padded_input_data)
    padded_label_data = np.stack(padded_label_data)

    padded_input_data = np.asarray(padded_input_data).astype(np.float32)
    padded_label_data =  np.asarray(padded_label_data).astype(np.int32)

    shuffled_indexes = np.random.permutation(padded_input_data.shape[0])
    padded_input_data = padded_input_data[shuffled_indexes]
    padded_label_data = padded_label_data[shuffled_indexes]

    np.load = np_load_old

    return padded_input_data, padded_label_data

def tfdata1(padded_input_data, label_data, batch_size, size):
    padded_input_data = padded_input_data[:size]
    label_data = label_data[:size]

    train1_data, valid1_data = np.split(
        padded_input_data, [np.shape(padded_input_data)[0] * 9 // 10])
    train1_label, valid1_label = np.split(
        label_data, [np.shape(label_data)[0] * 9 // 10])


    train1Data =  tf.data.Dataset.from_tensor_slices(
        (train1_data, train1_label)
    )
    train1Data = train1Data.shuffle(buffer_size=np.shape(train1_data)[0])
    train1Data = train1Data.batch(batch_size).prefetch(buffer_size=1)
    train1steps = np.shape(train1_data)[0] // batch_size


    valid1Data =  tf.data.Dataset.from_tensor_slices(
        (valid1_data, valid1_label)
    )
    valid1Data = valid1Data.shuffle(buffer_size=np.shape(valid1_data)[0])
    valid1Data = valid1Data.batch(batch_size).prefetch(buffer_size=1)
    valid1steps = np.shape(valid1_data)[0] // batch_size


    return train1Data,train1steps, \
           valid1Data, valid1steps


def getDataTrain2(max_length, data_split):
    np_load_old = partial(np.load)

    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    real_input = np.load('data/data_real_train.npy')
    real_label = np.load('data/label_real_train.npy')

    padded_input_data = []
    for _, v in enumerate(real_input):
        v = interpolate(int(max_length), v)
        padded_input_data.append(v)

    padded_input_data = np.stack(padded_input_data)
    label_data = np.stack(real_label)

    padded_input_data_ = np.asarray(padded_input_data).astype(np.float32)
    label_data_ = np.asarray(label_data).astype(np.int32)

    padded_input_data = padded_input_data_
    label_data = label_data_
    for i in range(5000):
        padded_input_data = np.concatenate((padded_input_data, padded_input_data_))
        label_data = np.concatenate((label_data, label_data_))

    shuffled_indexes = np.random.permutation(padded_input_data.shape[0])
    padded_input_data = padded_input_data[shuffled_indexes]
    label_data = label_data[shuffled_indexes]

    np.load = np_load_old
    return padded_input_data, label_data

def tfdata2(padded_input_data, label_data, batch_size, size, scale):
    padded_input_data = padded_input_data[:size]
    label_data = label_data[:size]

    if scale == 0.0:
        padded_input_data = padded_input_data
    else:
        padded_input_data = add_noise(padded_input_data, scale*1e-5)


    train2_data, valid2_data = np.split(
        padded_input_data, [np.shape(padded_input_data)[0] * 9 // 10])
    train2_label, valid2_label = np.split(
        label_data, [np.shape(label_data)[0] * 9 // 10])


    train2Data =  tf.data.Dataset.from_tensor_slices(
        (train2_data, train2_label)
    )
    train2Data = train2Data.shuffle(buffer_size=np.shape(train2_data)[0])
    train2Data = train2Data.batch(batch_size).prefetch(buffer_size=1)
    train2steps = np.shape(train2_data)[0] // batch_size

    valid2Data =  tf.data.Dataset.from_tensor_slices(
        (valid2_data, valid2_label)
    )
    valid2Data = valid2Data.shuffle(buffer_size=np.shape(valid2_data)[0])
    valid2Data = valid2Data.batch(batch_size).prefetch(buffer_size=1)
    valid2steps = np.shape(valid2_data)[0] // batch_size



    return train2Data,train2steps, valid2Data, valid2steps




def getDataTest(max_length, batch_size):
    np_load_old = partial(np.load)

    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    input_data = np.load('data/data_real_test.npy'.format(NN))
    label_data = np.load('data/label_real_test.npy'.format(NN))

    padded_input_data = []
    for _, v in enumerate(input_data):
        v = (interpolate(int(max_length), v))
        padded_input_data.append(v)
    test_data = np.asarray(padded_input_data).astype(np.float32)
    test_label = np.asarray(label_data).astype(np.int32)

    #
    # test_data = []
    # test_label = []
    # for i in range(40):
    #     test_data.append(padded_input_data)
    #     test_label.append(label_data)
    #
    #
    # test_data = np.stack(test_data)
    # test_label = np.stack(test_label)

    testData =  tf.data.Dataset.from_tensor_slices(
        (test_data, test_label)
    )
    testData = testData.batch(batch_size).prefetch(buffer_size=1)
    teststeps = np.shape(test_data)[0] // batch_size


    return testData, teststeps, test_data, test_label

