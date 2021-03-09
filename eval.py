from getData import getDataTrain1, getDataTrain2
from train import train, decode
import os
import numpy as np
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# size_list = [4000,16000,64000,128000]
size_list = [128000]
# size_list = [16000]

scale_list = [0.0]

# size_list = [160000]
# scale_list = [4]
label_pad = 25

tsteps_ascii = 7

max_length = label_pad*tsteps_ascii

input_dim = 2
num_classes = 28
learning_rate = 0.001
data_split = None
batch_size = 128
EPOCHS  = 15
load_model = False
SAVE_PATH = 'res/'
monitor = 'val_loss'




def main_fake(size):

    scale = ''
    y_true, y_pred = decode(SAVE_PATH, max_length, batch_size, scale, size)
    return y_true, y_pred

def main_real(size, scale):
    y_true, y_pred = decode(SAVE_PATH, max_length, batch_size, scale, size)
    return y_true, y_pred

def eval_save(fake, scale):
    y_true_fake = []
    y_pred_fake = []
    for size in size_list:
        if fake:
            y_true, y_pred = main_fake(size)
        else:
            y_true, y_pred = main_real(size, scale)

        y_true_fake.append(y_true)
        y_pred_fake.append(y_pred)


    y_true_fake = np.array(y_true_fake, dtype = 'object')
    y_pred_fake = np.array(y_pred_fake, dtype = 'object')

    if fake:
        np.save('y_true_fake.npy', y_true_fake)
        np.save('y_pred_fake.npy', y_pred_fake)
    else:
        np.save('y_true_real_{}.npy'.format(scale), y_true_fake)
        np.save('y_pred_real_{}.npy'.format(scale), y_pred_fake)

def eval_analysis(fake, scale):

    np_load_old = partial(np.load)

    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    if fake:
        y_true_fake = np.load('y_true_fake.npy')
        y_pred_fake = np.load('y_pred_fake.npy')
    else:
        y_true_fake = np.load('y_true_real_{}.npy'.format(scale))
        y_pred_fake = np.load('y_pred_real_{}.npy'.format(scale))

    np.load = np_load_old

    def to_dense(tensor):
        # tensor = tf.sparse.reset_shape(tensor, shape)
        tensor = tf.sparse.to_dense(tensor, default_value=0)
        tensor = tf.cast(tensor, tf.int32).numpy()
        tensor = np.transpose(tensor)
        tensor = tensor[~np.all(tensor == 0, axis=1)]
        decoded = tf.transpose(tensor)
        return decoded
    def accuracy_np(y_true, y_pred):
        num_errors = np.sum(np.not_equal(y_true, y_pred).astype(np.int32), axis=1)
        accuracy = 1 - np.mean(np.divide(num_errors, [5] * np.shape(y_true)[0]))
        return accuracy, num_errors

    accuracy_ = []
    num_errors_ = []
    for i in range(len(y_pred_fake)):
        y_true_ = y_true_fake[i]
        y_pred_ = to_dense(y_pred_fake[i][0])

        y_pred = []
        for v in y_pred_:
            residual = int(label_pad) - len(v)
            padding_array = np.zeros([int(residual)])
            y_pred.append(
                np.concatenate([v, padding_array], axis=0))
        y_pred = np.stack(y_pred).astype(np.int32)
        # y_pred = y_pred_
        y_true = []
        for v in y_true_:
            residual = int(label_pad) - len(v)
            padding_array = np.zeros([int(residual)])
            y_true.append(
                np.concatenate([v, padding_array], axis=0))
        y_true = np.stack(y_true).astype(np.int32)

        accuracy, num_errors = accuracy_np(y_true, y_pred)

        print('target: {}'.format(y_true))
        print('predict: {}'.format(y_pred))
        print('accuracy: {}'.format(accuracy))
        print('number of errors: {}'.format(num_errors))
        accuracy_.append(accuracy)
        num_errors_.append(num_errors)
    return accuracy_, num_errors_

def show_real():
    fake = False
    accuracy_ = []
    num_errors_ = []

    for scale in scale_list:
        eval_save(fake, scale)
        accuracy, num_errors = eval_analysis(fake, scale)
        accuracy_.append(accuracy)
        num_errors_.append(num_errors)

    print(num_errors_)
    print(accuracy_)
    return num_errors_, accuracy_
def show_fake():
    fake = True
    scale = ''
    eval_save(fake, scale)
    accuracy, num_errors = eval_analysis(fake, scale)
    print(accuracy)
    print(num_errors)
    return  num_errors, accuracy


def save():
    # num_errors_fake, accuracy_fake = show_fake()
    num_errors_real, accuracy_real = show_real()

    # np.savez('accuracy_res/fake', num_errors_fake, accuracy_fake)
    np.savez('accuracy_res/real', num_errors_real, accuracy_real)


    # print('synthetic data {}'.format(num_errors_fake))
    print('real data {}'.format(num_errors_real))

    # print('synthetic data {}'.format(accuracy_fake))
    print('real data {}'.format(accuracy_real))




def plot():
    save_path_fake = 'accuracy_res/fake.npz'
    save_path_real = 'accuracy_res/real.npz'

    npzfile_fake = np.load(save_path_fake)
    npzfile_real = np.load(save_path_real)

    num_errors_fake = npzfile_fake['arr_0']
    accuracy_fake = npzfile_fake['arr_1']

    num_errors_real = npzfile_real['arr_0']
    accuracy_real = npzfile_real['arr_1']

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(25, 6))

    ax.plot(size_list, accuracy_fake,linestyle='-', marker='o', linewidth=5,markersize=17, label = "Synthetic Data")

    for i in range(len(accuracy_real)):
        # if i == 0:
        #     continue
        ax.plot(size_list, accuracy_real[i],linestyle='-', marker='o', linewidth=5,markersize=12, label = "Gaussian Noise {}e-5".format(scale_list[i]))

    ax.set_xlabel('Data Size')
    # Set the y axis label of the current axis.
    ax.set_ylabel('Accuracy')

    ax.set_xscale('log')

    # ax.legend()
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # plt.show()
    plt.savefig('accuracy_res/accuracy_plot.pdf', bbox_inches = 'tight',
    pad_inches = 0)




if __name__ == "__main__":
    save()
    # plot()

