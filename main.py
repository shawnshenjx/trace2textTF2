from getData import getDataTrain1, getDataTrain2
from train import train, decode
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.autograph.set_verbosity(0)

size_list = [4000,8000,16000,32000,64000,128000]
scale_list = [0, 2, 4]
label_pad = 25
tsteps_ascii = 7
multipler = 1000
max_length = label_pad*tsteps_ascii
input_dim = 2
num_classes = 28
learning_rate = 0.001
data_split = None
batch_size = 128
EPOCHS  = 20
load_model = False
SAVE_PATH = 'res/'
monitor = 'val_loss'
restore = True

# def main_fake(data, size):
#
#     scale = ''
#     train(max_length, input_dim, num_classes, learning_rate, data,
#      batch_size, size, scale, EPOCHS, SAVE_PATH, monitor, load_model,
#      restore = False)


def main_real(size, scale):
    data, number_train, number_valid  = getDataTrain2(multipler)

    train(max_length, input_dim, num_classes, learning_rate,data,
          batch_size, size, scale, EPOCHS, SAVE_PATH, monitor, load_model,
          restore, number_train, number_valid)

if __name__ == "__main__":
    main_real(128000,0)