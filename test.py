import tensorflow as tf
from model import rnn_att_model, CTCLoss
from tensorflow import keras
from metrics import EditDistance, SequenceAccuracy
from getData import tfdata1, tfdata2, getDataTest
import pickle
import numpy as np

from functools import partial

size_list = [1000,8000,16000,32000,64000,128000]
noise_list = [0,2,4,8]
scale_list = [0, 2, 4]
data_split_list = ['71', '62', '53', '17']

max_length = 800
input_dim = 18
num_classes = 12
learning_rate = 0.001
data_split = 128
batch_size = 64
EPOCHS  = 15
load_model = False
SAVE_PATH = 'res'
monitor = 'val_loss'


# decoder = decoders.CTCGreedyDecoder(table_path)
Data, steps, data_test, y_true = getDataTest(max_length, batch_size)
model = keras.models.load_model(SAVE_PATH, custom_objects={'tf': tf, 'CTCLoss': CTCLoss, 'EditDistance':EditDistance})
outputs = model.predict(Data, verbose=1)
sequence_length = tf.fill([tf.shape(outputs)[0]], tf.shape(outputs)[1])

decoded, _ = tf.nn.ctc_greedy_decoder(
    tf.transpose(outputs, perm = [1,0,2]), sequence_length, merge_repeated=True
)

def to_dense(tensor):
    # tensor = tf.sparse.reset_shape(tensor, shape)
    tensor = tf.sparse.to_dense(tensor, default_value=0)
    tensor = tf.cast(tensor, tf.int32).numpy()
    tensor = np.transpose(tensor)
    tensor = tensor[~np.all(tensor == 0, axis=1)]
    decoded = tf.transpose(tensor)
    return decoded

y_pred = to_dense(decoded[0])

num_errors = tf.math.reduce_sum(
    tf.cast(tf.math.not_equal(y_true, y_pred), tf.int32), axis=1)
num_errors = num_errors.numpy()
accuracy = 1 - np.mean(np.divide(num_errors, [5] * np.shape(y_true)[0]))


# np.save('{}outputs{}_{}.npy', outputs)
# np.save('{}decoded{}_{}.npy', decoded)


np_load_old = partial(np.load)
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

y_true = np.load('data/label_real_test.npy')

decoded = np.load('decoded.npy')

np.load = np_load_old


def to_dense(tensor):
    # tensor = tf.sparse.reset_shape(tensor, shape)
    tensor = tf.sparse.to_dense(tensor, default_value=0)
    tensor = tf.cast(tensor, tf.int32).numpy()
    tensor = np.transpose(tensor)
    tensor = tensor[~np.all(tensor == 0, axis=1)]
    decoded = tf.transpose(tensor)
    return decoded


y_pred = to_dense(decoded[0])

# num_errors = tf.math.reduce_any(
#     tf.math.not_equal(y_true, y_pred), axis=1)

num_errors = tf.math.reduce_sum(
    tf.cast(tf.math.not_equal(y_true, y_pred),tf.int32), axis=1)
num_errors = num_errors.numpy()
accuracy =1- np.mean(np.divide(num_errors,[5]*np.shape(y_true)[0]))
# num_errors = tf.reduce_sum(num_errors)
print(accuracy)
print(y_pred)
print(y_true)

