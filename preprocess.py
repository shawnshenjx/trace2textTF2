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
from utils import denoise, interpolate, normalize, add_features, get_ascii_sequences, get_stroke_sequence, l2_norm, fake_generator_saver
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import math

data_type = 'real'
if data_type == 'real':
    path = 'processed_holo2_mixed/*.csv'
elif data_type == 'fake':
    path = '/home/shawn/desktop/GAN_DE/20210213-150805/res/save_holo/*.npy'

max_c_length = 25
tsteps_ascii = 7



# def fake_main():
#
#     text_line_data_all = []
#     label_text_line_all = []
#
#     for i, fname in enumerate(sorted(glob.glob(path), reverse=False)):
#         print(fname)
#         ############# label data #############
#
#         phrases = fname.split('.')[-2]
#         phrases = phrases.split('/')[-1]
#         phrases = phrases.split('_')[1:-1]
#         phrases = ' '.join(phrases)
#         print(phrases)
#         sequence = np.load(fname)
#
#         # sequence = interpolate(sequence, len(phrases) * 50)
#
#         sequence = normalize(sequence)
#         sequence = add_features(sequence)
#
#
#         label_text_line = get_ascii_sequences(phrases)
#         print(get_ascii_sequences(phrases))
#         text_line_data_all.append(sequence)
#         label_text_line_all.append(label_text_line)
#
#
#     text_line_data_all = np.array(text_line_data_all, dtype='object')
#     label_text_line_all = np.array(label_text_line_all, dtype='object')
#
#     print('input data with shape {}'.format(np.shape(text_line_data_all)))
#     print('label data with shape {}'.format(np.shape(label_text_line_all)))
#
#     np.save("data/data_{}".format(data_type), text_line_data_all)
#     np.save("data/label_{}".format(data_type), label_text_line_all)
#     print("Successfully saved!")



def real_main():
    text_line_data_all = []

    label_text_line_all = []

    char_len = []
    stroke_len = []
    dist_len = []
    open('real_phrases.txt', 'w').close()

    with open('real_phrases_holo.txt', 'w') as f:
        for i, fname in enumerate(sorted(glob.glob(path), reverse=False)):
            ############# label data #############
            phrases = (fname.split('.')[-2]).split('/')[-1]
            phrases = phrases.split('_')[1:]
            phrases = '_'.join(phrases)
            print(phrases)
            f.write("%s \n" % phrases)

            coords, coords_ = get_stroke_sequence(fname, phrases, max_c_length, tsteps_ascii)
            fake_coords = fake_generator_saver(phrases, max_c_length, tsteps_ascii)
            dist = l2_norm(phrases, tsteps_ascii, coords_[:,:2], fake_coords)
            if dist > 0.0055:
                continue
            coords[:,:2] = normalize(coords[:,:2])
            print(np.shape(coords))
            label_text_line = get_ascii_sequences(phrases, max_c_length)
            print(get_ascii_sequences(phrases, max_c_length))
            text_line_data_all.append(coords)
            label_text_line_all.append(label_text_line)
            dist_len.append(dist)
            char_len.append(len(phrases))
            stroke_len.append(len(coords))

    text_line_data_all = np.array(text_line_data_all, dtype = 'object')
    label_text_line_all = np.array(label_text_line_all, dtype = 'object')

    # save as .npy
    print('input data with shape {}'.format(np.shape(text_line_data_all)))
    print('label data with shape {}'.format(np.shape(label_text_line_all)))

    print('average phrase length {}, max length {}, min length {}'.format(np.mean(char_len), np.max(char_len),
                                                                          np.min(char_len)))
    print('average data length {}, max length {}, min length {}'.format(np.mean(stroke_len), np.max(stroke_len),
                                                                        np.min(stroke_len)))

    np.save("data/data_real_train", text_line_data_all)
    np.save("data/label_real_train", label_text_line_all)
    print("Successfully saved!")


if __name__ == "__main__":
    real_main()

