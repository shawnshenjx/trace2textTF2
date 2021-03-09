import numpy as np
from scipy.interpolate import interp1d
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from matplotlib import pyplot
import glob
from functools import partial
import os
import shutil
import seaborn as sns
from math import log2
from collections import defaultdict
import scipy
import pandas as pd
# from sklearn.preprocessing import normalize


matplotlib.rcParams.update({'font.size': 22})


alphabet = [' ', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
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
    padded_seq = np.concatenate(([sequence[0]], [sequence[0]], [sequence[0]],sequence, [sequence[-1]], [sequence[-1]], [sequence[-1]]), axis=0)
    aspect = np.zeros(len(sequence))
    slope = np.zeros((len(sequence), 2))
    curliness = np.zeros(len(sequence))
    linearity = np.zeros(len(sequence))
    for j in range(3, len(sequence)+3):
        vicinity = np.asarray([padded_seq[j-3],padded_seq[j-2], padded_seq[j-1], padded_seq[j], padded_seq[j+1], padded_seq[j+2],padded_seq[j+3]])
        delta_x = max(vicinity[:, 0]) - min(vicinity[:, 0])
        delta_y = max(vicinity[:, 1]) - min(vicinity[:, 1])

        # delta_x = vicinity[-1, 0] - vicinity[0, 0]
        # delta_y = vicinity[-1, 1] - vicinity[0, 1]
        slope_vec = vicinity[-1] - vicinity[0]

        #aspect of trajectory
        aspect[j-3] = (delta_y - delta_x) / (delta_y + delta_x)

        #cos and sin of slope_angle of straight line from vicinity[0] to vicinity[-1]
        slope_angle = np.arctan(np.abs(np.divide(slope_vec[1], slope_vec[0]))) * np.sign(np.divide(slope_vec[1], slope_vec[0]))
        slope[j-3] = [np.cos(slope_angle), np.sin(slope_angle)]

        #length of trajectory divided by max(delta_x, delta_y)
        curliness[j-3] = np.sum([np.linalg.norm(vicinity[k+1] - vicinity[k], 2) for k in range(len(vicinity)-1)]) / max(delta_x, delta_y)

        #avg squared distance from each point to straight line from vicinity[0] to vicinity[-1]
        linearity[j-3] = np.mean([np.power(np.divide(np.cross(slope_vec, vicinity[0] - point), np.linalg.norm(slope_vec, 1)), 2) for point in vicinity])

    vicinity_features = np.column_stack((curliness))
    curliness = np.expand_dims(curliness, axis=1)


    #     result = np.nan_to_num(np.concatenate((offsets, gradient, curvature, vicinity_features), axis=1)).tolist()
    result = np.nan_to_num(np.concatenate((sequence, curvature, curliness), axis=1)).tolist()
    result = np.array(result)


    return result


def get_ascii_sequences(phrases,max_c_length):
    lines = encode_ascii(phrases,max_c_length)
    # lines = encode_ascii(phrases)
    return lines


def l2_norm(phrases,tsteps_ascii,text_line_data, fake_coords):
    dist1 = np.linalg.norm(text_line_data[:,0]-fake_coords[:,0])
    dist2 = np.linalg.norm(text_line_data[:,1]-fake_coords[:,1])
    dist = (dist1+dist2)/(2*len(phrases)*tsteps_ascii)
    return dist


def get_stroke_sequence(fname, phrase, max_c_length, tsteps_asii):
    max_x_length = max_c_length * tsteps_asii
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
    coords_ = interpolate(coords, len(phrase), tsteps_asii)

    # coords = add_features(coords)
    return coords, coords_[:max_x_length]



def encode_ascii(ascii_string,max_c_length):
    """
    encodes ascii string to array of ints
    """
    return np.array(list(map(lambda x: alpha_to_num[x], ascii_string)))[:max_c_length]


def interpolate(stroke, len_ascii, tsteps_asii):
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



def normalize(coords):
    """
    normalizes strokes to median unit norm
    """
    coords_ = np.copy(coords)
    coords_[:, :] /= np.median(np.linalg.norm(coords[:, :], axis=1))
    return coords_

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




def key_convert2csv():
    keys = []
    x_pos = []
    y_pos = []
    f = open('holokeyboard.txt', 'r')
    str = f.readline()
    str = f.readline()
    while len(str) > 1:
        info = str[:-1].split(';')
        keys.append(info[0])
        x_pos.append(int(info[1]))
        y_pos.append(int(info[2]))

        str = f.readline()

    df = pd.DataFrame({
        'keys': keys,
        'x_pos': x_pos,
        'y_pos': y_pos,
    })

    df.to_csv('new_holokeyboard.csv')

    return df


def interpolate_linear(stroke, len_ascii, tsteps_ascii):
    xy_coords = stroke

    if len(stroke) > 3:
        f_x = interp1d(np.arange(len(stroke)), stroke[:, 0])
        f_y = interp1d(np.arange(len(stroke)), stroke[:, 1])

        xx = np.linspace(0, len(stroke) - 1, len_ascii * tsteps_ascii)
        yy = np.linspace(0, len(stroke) - 1, len_ascii * tsteps_ascii)

        x_new = f_x(xx)
        y_new = f_y(yy)

        xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

    return xy_coords


def fake_generator(fake_str, max_x_length, tsteps_ascii):
    df = pd.read_csv("new_holokeyboard.csv", index_col='keys')

    scale = 0.0002

    stroke = []
    for j in fake_str:
        stroke.append([df.loc[j][1], df.loc[j][2]])
    stroke_np = np.reshape(np.array(stroke) * scale, (-1, 2))

    coords = interpolate_linear(stroke_np, len(fake_str), tsteps_ascii=tsteps_ascii)

    coords = np.transpose(coords)
    #     coords = normalize(coords[:max_x_length])
    return coords[:max_x_length]
    # return stroke_np[:,0], stroke_np[:,1], stroke_fi[:,0], stroke_fi[:,1]


def fake_generator_saver(phrase, max_c_length, tsteps_ascii):
    max_x_length = max_c_length * tsteps_ascii
    x_label = encode_ascii(phrase, max_c_length)
    x = fake_generator(phrase, max_x_length, tsteps_ascii)
    x_label = np.array(x_label)
    x = np.squeeze(x)
    x = np.transpose(x)
    return x



