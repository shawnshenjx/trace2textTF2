import matplotlib.pyplot as plt
import numpy as np
import itertools
plt.switch_backend('agg')
from getData import getDataTrain1, getDataTrain2
from model import rnn_att_model
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pickle


def plot_loss(hist, SAVE_PATH, indx):
    sns.set()

    acc = hist['sparse_categorical_accuracy']
    val_acc = hist['val_sparse_categorical_accuracy']

    loss = hist['loss']
    val_loss = hist['val_loss']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.plot(loss, label='train')
    ax1.plot(val_loss, label='validation')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title('Model loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(acc, label='train')
    ax2.plot(val_acc, label='validation')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title('Model accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(SAVE_PATH + 'loss_{}.png'.format(indx))
    plt.close()

