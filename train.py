import tensorflow as tf
from model import rnn_att_model, CTCLoss
from tensorflow import keras
from metrics import EditDistance, SequenceAccuracy
from getData import tfdata1, tfdata2, getDataTest
import pickle
import numpy as np
import os
from plot import plot_loss
import decoders

def train(input_shape1, input_shape2, num_classes, learning_rate,input1, label1,
          input2, label2, batch_size, size, scale, EPOCHS, SAVE_PATH, monitor, load_model):
    if scale != '':
        train1Data, train1steps, \
        valid1Data, valid1steps \
            = tfdata2(input2, label2, batch_size, size, scale)
    else:
        train1Data, train1steps, \
        valid1Data, valid1steps, \
            = tfdata1(input1, label1, batch_size, size)

    SAVE_PATH = SAVE_PATH + '{}_{}/'.format(scale, size)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if load_model:
        model = keras.models.load_model(SAVE_PATH,  custom_objects={'tf': tf,
                                                                    'CTCLoss': CTCLoss, 'EditDistance':EditDistance})
        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss=CTCLoss(), metrics=[EditDistance()])
        model.summary()
    else:
        model = rnn_att_model(input_shape1, input_shape2, num_classes)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss=CTCLoss(), metrics=[EditDistance()])
        model.summary()

    # Stop if the validation accuracy doesn't imporove for x epochs
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=3, verbose=1)
    # Reduce LR on Plateau
    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=2, verbose=1)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=SAVE_PATH,
        save_weights_only=False,
        monitor = monitor,
        mode='min',
        save_freq = int(train1steps*3),
        save_best_only=True)

    history_lstm = model.fit(train1Data.repeat(),
                             steps_per_epoch=train1steps,
                             validation_data=valid1Data.repeat(),
                             validation_steps=valid1steps,
                             epochs=EPOCHS,
                             callbacks=[earlyStopping, reduceLR, model_checkpoint_callback])

    ## Save model

    model.save(SAVE_PATH)

    # serialize weights to HDF5
    model.save_weights(SAVE_PATH + "weights")
    print("Saved lstm model to res")

    ## Save history data
    with open(SAVE_PATH + "train_results.pickle", "wb") as handle:
        pickle.dump(history_lstm.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved lstm training history to res")


    # with open('res/train_results.pickle', 'rb') as handle:
    #     history = pickle.load(handle)


def decode(SAVE_PATH, max_length, batch_size, scale, size):
    # decoder = decoders.CTCGreedyDecoder(table_path)
    Data, steps, data_test, y_true = getDataTest(max_length, batch_size)
    model = keras.models.load_model(SAVE_PATH+'{}_{}/'.format(scale,size), custom_objects={'tf': tf, 'CTCLoss': CTCLoss, 'EditDistance':EditDistance})
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

    real_str_decoded = ''.join(
        [letter_table[x] for x in np.asarray(real_predict.values[:real_target_length])])


    return accuracy

    # np.save('{}outputs{}_{}.npy', outputs)
    # np.save('{}decoded{}_{}.npy', decoded)