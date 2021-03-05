from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os
import csv
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import re
import math
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import math

data_type = 'fake'
if data_type == 'real':
    path = 'processed_real_all/*.csv'
elif data_type == 'holo':
    path= 'data/data/holo_real/*.csv'
elif data_type == 'fake':
    path = '/home/shawn/desktop/GAN_DE/20210213-150805/res/save_holo/*.npy'


alphabet = ['_', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z']


alphabet_ord = list(map(ord, alphabet))
alpha_to_num = defaultdict(int, list(map(reversed, enumerate(alphabet))))
num_to_alpha = dict(enumerate(alphabet_ord))



def add_features(sequence):
    sequence = np.asarray(sequence)
    next_seq = np.append(sequence[1:, :], [sequence[-1, :]], axis=0)
    prev_seq = np.append([sequence[0, :]], sequence[:-1, :], axis=0)

    # compute gradient
    gradient = np.subtract(sequence, prev_seq)

    #compute curvature
    vec_1 = np.multiply(gradient, -1)
    vec_2 = np.subtract(next_seq, sequence)
    angle = np.divide(np.sum(vec_1*vec_2, axis=1),
                      np.linalg.norm(vec_1, 2, axis=1)*np.linalg.norm(vec_2, 2, axis=1))

    angle[np.isnan(angle)]=0

    curvature = np.column_stack((np.cos(angle), np.sin(angle)))

    #compute vicinity (5-points) - curliness/linearity
    padded_seq = np.concatenate(([sequence[0]], [sequence[0]], sequence, [sequence[-1]], [sequence[-1]]), axis=0)
    aspect = np.zeros(len(sequence))
    slope = np.zeros((len(sequence), 2))
    curliness = np.zeros(len(sequence))
    linearity = np.zeros(len(sequence))
    for j in range(2, len(sequence)+2):
        vicinity = np.asarray([padded_seq[j-2], padded_seq[j-1], padded_seq[j], padded_seq[j+1], padded_seq[j+2]])
        delta_x = max(vicinity[:, 0]) - min(vicinity[:, 0])
        delta_y = max(vicinity[:, 1]) - min(vicinity[:, 1])

        # delta_x = vicinity[-1, 0] - vicinity[0, 0]
        # delta_y = vicinity[-1, 1] - vicinity[0, 1]
        slope_vec = vicinity[-1] - vicinity[0]

        #aspect of trajectory
        aspect[j-2] = (delta_y - delta_x) / (delta_y + delta_x)

        #cos and sin of slope_angle of straight line from vicinity[0] to vicinity[-1]
        slope_angle = np.arctan(np.abs(np.divide(slope_vec[1], slope_vec[0]))) * np.sign(np.divide(slope_vec[1], slope_vec[0]))
        slope[j-2] = [np.cos(slope_angle), np.sin(slope_angle)]

        #length of trajectory divided by max(delta_x, delta_y)
        curliness[j-2] = np.sum([np.linalg.norm(vicinity[k+1] - vicinity[k], 2) for k in range(len(vicinity)-1)]) / max(delta_x, delta_y)

        #avg squared distance from each point to straight line from vicinity[0] to vicinity[-1]
        linearity[j-2] = np.mean([np.power(np.divide(np.cross(slope_vec, vicinity[0] - point), np.linalg.norm(slope_vec, 1)), 2) for point in vicinity])

    vicinity_features = np.column_stack((aspect, slope, curliness, linearity))

    # add features to signal
    offsets = coords_to_offsets(sequence)

    result = np.nan_to_num(np.concatenate((sequence, offsets, gradient, curvature, vicinity_features), axis=1)).tolist()

    return result

def denoise(coords):
    """
    smoothing filter to mitigate some artifacts of the data collection
    """


    x_new = savgol_filter(coords[:, 0], 7, 3, mode='nearest')
    y_new = savgol_filter(coords[:, 1], 7, 3, mode='nearest')
    stroke = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

    return stroke

def add_noise(coords, scale=0.05):
    """
    adds gaussian noise to strokes
    """
    coords = np.copy(coords)
    coords[1:, :] += np.random.normal(loc=0.0, scale=scale, size=coords[1:, :].shape)
    return coords



def encode_ascii(ascii_string):
    """
    encodes ascii string to array of ints
    """
    return np.array(list(map(lambda x: alpha_to_num[x], ascii_string)) )

# def encode_ascii(ascii_string):
#     """
#     encodes ascii string to array of ints
#     """
#     ascii_string_bi=[ascii_string[i]+ascii_string[i+1] for i in range(len(ascii_string)-1)]
#     return np.array(list(map(lambda x: alpha_to_num[x], ascii_string_bi)) + [0])

def normalize(offsets):
    """
    normalizes strokes to median unit norm
    """
    offsets = np.copy(offsets)
    offsets[:, :2] /= np.median(np.linalg.norm(offsets[:, :2], axis=1))
    return offsets


def coords_to_offsets(coords):
    """
    convert from coordinates to offsets
    """
    offsets = coords[1:, :2] - coords[:-1, :2]
    offsets = np.concatenate([np.array([[0, 0]]), offsets], axis=0)
    # offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    # offsets = np.concatenate([np.array([[0, 0, 0]]), offsets], axis=0)
    return offsets


def offsets_to_coords(offsets):
    """
    convert from offsets to coordinates
    """
    return np.cumsum(offsets[:, :2], axis=0)


def interpolate(stroke, len_ascii, tsteps_asii=20):
    """
    interpolates strokes using cubic spline
    """

    xy_coords = stroke[:, :2]

    if len(stroke) > 3:
        f_x = interp1d(np.arange(len(stroke)), stroke[:, 0], kind='cubic')
        f_y = interp1d(np.arange(len(stroke)), stroke[:, 1], kind='cubic')

        xx = np.linspace(0, len(stroke) - 1, len_ascii*tsteps_asii)
        yy = np.linspace(0, len(stroke) - 1, len_ascii*tsteps_asii)

        x_new = f_x(xx)
        y_new = f_y(yy)

        xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

    return xy_coords



def get_ascii_sequences(phrases):
    lines = encode_ascii(phrases)
    # lines = encode_ascii(phrases)
    return lines




def get_stroke_sequence(fname,phrase=None):

    coords = []

    with open(fname, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row == []:
                continue

            if row[0] == '':
                continue

            coords.append([float(row[0]),
                           float(row[1]),
            ])
    coords = np.array(coords)
    coords = np.reshape(coords, [-1, 2])

    coords = denoise(coords)
    coords = interpolate(coords, len(phrase) * 100)
    coords = interpolate(coords, len(phrase))

    # offsets = coords
    # offsets = coords_to_offsets(coords)
    coords = normalize(coords)
    coords = add_features(coords)
    return coords


def collect_data():

    stroke_fnames=[]

    for fname in sorted(glob.glob(path), reverse=True):
        stroke_fnames.append(fname)
    return stroke_fnames

def fake_main():

    text_line_data_all = []
    label_text_line_all = []

    for i, fname in enumerate(sorted(glob.glob(path), reverse=False)):
        print(fname)
        ############# label data #############

        phrases = fname.split('.')[-2]
        phrases = phrases.split('/')[-1]
        phrases = phrases.split('_')[1:-1]
        phrases = ' '.join(phrases)
        print(phrases)
        sequence = np.load(fname)

        # sequence = interpolate(sequence, len(phrases) * 50)

        sequence = normalize(sequence)
        sequence = add_features(sequence)


        label_text_line = get_ascii_sequences(phrases)
        print(get_ascii_sequences(phrases))
        text_line_data_all.append(sequence)
        label_text_line_all.append(label_text_line)


    text_line_data_all = np.array(text_line_data_all, dtype='object')
    label_text_line_all = np.array(label_text_line_all, dtype='object')

    print('input data with shape {}'.format(np.shape(text_line_data_all)))
    print('label data with shape {}'.format(np.shape(label_text_line_all)))

    np.save("data/data_{}".format(data_type), text_line_data_all)
    np.save("data/label_{}".format(data_type), label_text_line_all)
    print("Successfully saved!")



def real_main():
    text_line_data_all = []

    label_text_line_all = []

    char_len = []
    stroke_len = []
    open('real_phrases.txt', 'w').close()

    with open('real_phrases.txt', 'w') as f:
        for i, fname in enumerate(sorted(glob.glob(path), reverse=False)):
            ############# label data #############
            print(fname)

            phrases = (fname.split('.')[-2]).split('/')[-1]
            phrases = phrases.split('_')[1:]
            phrases = '_'.join(phrases)
            sequence = get_stroke_sequence(fname, phrases)
            print(phrases)

            f.write("%s \n" % phrases)

            text_line_data=sequence

            label_text_line = get_ascii_sequences(phrases)


            text_line_data_all.append(text_line_data)
            label_text_line_all.append(label_text_line)

            char_len.append(len(phrases))
            stroke_len.append(len(text_line_data_all))


    text_line_data_all = np.array(text_line_data_all, dtype = 'object')
    label_text_line_all = np.array(label_text_line_all, dtype = 'object')

    # save as .npy
    print('input data with shape {}'.format(np.shape(text_line_data_all)))
    print('label data with shape {}'.format(np.shape(label_text_line_all)))

    np.save("data/data_{}".format(data_type), text_line_data_all)
    np.save("data/label_{}".format(data_type), label_text_line_all)
    print("Successfully saved!")

def holo_main():
    text_line_data_all = []

    label_text_line_all = []

    char_len = []
    stroke_len = []
    open('real_phrases_holo.txt', 'w').close()

    with open('real_phrases_holo.txt', 'w') as f:
        for i, fname in enumerate(sorted(glob.glob(path), reverse=False)):
            ############# label data #############
            print(fname)

            phrases = (fname.split('.')[-2]).split('/')[-1]
            phrases = phrases.split('_')[1:]
            phrases = '_'.join(phrases)
            sequence = get_stroke_sequence(fname, phrases)
            print(phrases)

            f.write("%s \n" % phrases)

            text_line_data=sequence

            label_text_line = get_ascii_sequences(phrases)


            text_line_data_all.append(text_line_data)
            label_text_line_all.append(label_text_line)

            char_len.append(len(phrases))
            stroke_len.append(len(text_line_data_all))


    text_line_data_all = np.array(text_line_data_all, dtype = 'object')
    label_text_line_all = np.array(label_text_line_all, dtype = 'object')

    # save as .npy
    print('input data with shape {}'.format(np.shape(text_line_data_all)))
    print('label data with shape {}'.format(np.shape(label_text_line_all)))

    np.save("data/data_{}".format(data_type), text_line_data_all)
    np.save("data/label_{}".format(data_type), label_text_line_all)
    print("Successfully saved!")

if __name__ == "__main__":
    if data_type =='fake':
        fake_main()
    elif data_type =='real':
        real_main()
    elif data_type =='holo':
        holo_main()
