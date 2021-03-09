import os
import pandas as pd
from collections import defaultdict
import numpy as np
import glob
import tensorflow as tf
from scipy.interpolate import interp1d
from functools import partial
from utils import interpolate, add_noise





def getDataTrain1(max_length, label_pad, tsteps_asii_):
    np_load_old = partial(np.load)

    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    input_data = np.load('data/data_fake.npy')
    label_data = np.load('data/label_fake.npy')


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

    return [padded_input_data, padded_label_data]

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

    valid1Data =  tf.data.Dataset.from_tensor_slices(
        (valid1_data, valid1_label)
    )
    valid1Data = valid1Data.shuffle(buffer_size=np.shape(valid1_data)[0])
    valid1Data = valid1Data.batch(batch_size).prefetch(buffer_size=1)

    return train1Data,valid1Data

def multiple(padded_input_data_,label_data_,multipler):
    padded_input_data = padded_input_data_
    label_data = label_data_
    for i in range(multipler):
        padded_input_data = np.concatenate((padded_input_data, padded_input_data_))
        label_data = np.concatenate((label_data, label_data_))


    return padded_input_data, label_data

def shuffle(padded_input_data, padded_label_data):
    shuffled_indexes = np.random.permutation(padded_input_data.shape[0])
    padded_input_data = padded_input_data[shuffled_indexes]
    padded_label_data = padded_label_data[shuffled_indexes]
    return padded_input_data, padded_label_data

def getDataTrain2(multipler):
    np_load_old = partial(np.load)
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    input_data = np.load('data/data_real_train.npy')
    label_data = np.load('data/label_real_train.npy')
    np.load = np_load_old

    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, padding = 'post')
    label_data = tf.keras.preprocessing.sequence.pad_sequences(label_data, padding = 'post')

    padded_input_data, padded_label_data = shuffle(input_data, label_data)

    train2_data, valid2_data = np.split(
        padded_input_data, [np.shape(padded_input_data)[0] * 9 // 10])
    train2_label, valid2_label = np.split(
        padded_label_data, [np.shape(padded_label_data)[0] * 9 // 10])

    np.save('data/data_real_test.npy',valid2_data)
    np.save('data/label_real_test.npy', valid2_label)

    train2_data, train2_label = multiple(train2_data, train2_label,multipler = multipler)
    valid2_data, valid2_label = multiple(valid2_data, valid2_label, multipler=multipler)

    train2_data, train2_label = shuffle(train2_data, train2_label)
    valid2_data, valid2_label = shuffle(valid2_data, valid2_label)
    number_train = len(train2_data)
    number_valid = len(train2_data)
    return [train2_data, train2_label, valid2_data, valid2_label] , number_train, number_valid

def tfdata2(train2_data, train2_label,valid2_data, valid2_label, batch_size, size, scale):
    train2_data = train2_data[:size]
    train2_label = train2_label[:size]
    valid2_data = valid2_data[:size]
    valid2_label = valid2_label[:size]

    if scale == 0:
        train2_data = train2_data
    else:
        train2_data = add_noise(train2_data, scale*1e-5)


    train2Data =  tf.data.Dataset.from_tensor_slices(
        (train2_data, train2_label)
    )
    train2Data = train2Data.shuffle(buffer_size=np.shape(train2_data)[0])
    train2Data = train2Data.batch(batch_size).prefetch(buffer_size=1)

    valid2Data =  tf.data.Dataset.from_tensor_slices(
        (valid2_data, valid2_label)
    )
    valid2Data = valid2Data.shuffle(buffer_size=np.shape(valid2_data)[0])
    valid2Data = valid2Data.batch(batch_size).prefetch(buffer_size=1)

    return train2Data, valid2Data


def getDataTest(batch_size):
    np_load_old = partial(np.load)
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    input_data = np.load('data/data_real_test.npy')
    label_data = np.load('data/label_real_test.npy')
    np.load = np_load_old
    testData =  tf.data.Dataset.from_tensor_slices(
        (input_data, label_data)
    )
    testData = testData.batch(batch_size).prefetch(buffer_size=1)

    return input_data, label_data

