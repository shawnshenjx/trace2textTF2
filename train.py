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
import time
import logging
from getData import getDataTrain2
loss_object = CTCLoss()

letter_table = [' ', '_',  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]/max(len(s1), len(s2))


def to_dense(tensor):
    # tensor = tf.sparse.reset_shape(tensor, shape)
    tensor = tf.sparse.to_dense(tensor, default_value=0)
    tensor = tf.cast(tensor, tf.int32).numpy()
    # tensor = np.transpose(tensor)
    # tensor = tensor[~np.all(tensor == 0, axis=1)]
    # decoded = tf.transpose(tensor)
    return tensor

def init_logging(log_dir):
    logging_level = logging.INFO

    log_file = 'log_valid.txt'

    log_file = os.path.join(log_dir, log_file)
    if os.path.isfile(log_file):
        os.remove(log_file)

    logging.basicConfig(
        filename=log_file,
        level=logging_level,
        format='[[%(asctime)s]] %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    return logging



def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train_step(epoch, train1Data,model, optimizer,epoch_loss_avg, epoch_accuracy, log):
    print('---------------------------- Start Train Epoch {} ---------------------------------\n'.format(epoch))
    for step, (x_batch_train, y_batch_train) in enumerate(train1Data):
        loss_value, grads = grad(model, x_batch_train, y_batch_train)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss_avg.update_state(loss_value)
        epoch_accuracy.update_state(y_batch_train, model(x_batch_train, training=True))

        if step % 50 == 0:
            log.info("Step {:03d}: Loss: {:.3f}, Editdistance: {:.3}".format(step,
                                                                             epoch_loss_avg.result(),
                                                                             epoch_accuracy.result()))
    print('---------------------------- Finish Train Epoch {} ---------------------------------\n'.format(epoch))


def test(batch_size,model, log):
    input_data, label_data = getDataTest(batch_size)
    val_logits = model(input_data)
    for i in range(len(label_data)):
        if i % 5 == 0:
            target = ''.join(
                [letter_table[x] for x in label_data[i]])

            sequence_length = tf.fill([tf.shape(val_logits)[0]], tf.shape(val_logits)[1])
            decoded_gd, _ = tf.nn.ctc_greedy_decoder(
                tf.transpose(val_logits, perm=[1, 0, 2]), sequence_length, merge_repeated=True
            )
            dense_decoded = to_dense(decoded_gd[0])
            predict_gd = ''.join(
                [letter_table[x] for x in dense_decoded[i]])

            decoded_bs, _ = tf.nn.ctc_beam_search_decoder(
                tf.transpose(val_logits, perm=[1, 0, 2]), sequence_length, beam_width=10)
            dense_decoded = to_dense(decoded_bs[0])
            predict_bs = ''.join(
                [letter_table[x] for x in dense_decoded[i]])

            target = target.split(' ')
            while '' in target:
                target.remove('')
            target = ''.join(target)

            predict_gd = predict_gd.split(' ')
            while '' in predict_gd:
                predict_gd.remove('')
            predict_gd = ''.join(predict_gd)

            # predict_bs = predict_bs.split(' ')
            # while '' in predict_bs:
            #     predict_bs.remove('')
            # predict_bs = ''.join(predict_bs)

            dist = levenshteinDistance(target, predict_gd)

            log.info('Target                : "{}"'.format(target))
            log.info('Predict Greedy Search : "{}"'.format(predict_gd))
            log.info('Levenshtein Distance  :  {}\n'.format(dist))
            # log.info('Predict Beam Search   : "{}"\n'.format(predict_bs))

def validation(valid1Data,model, val_accuracy, log, start_time):
    for x_batch_val, y_batch_val in valid1Data:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_accuracy.update_state(y_batch_val, val_logits)
    val_acc = val_accuracy.result()
    val_accuracy.reset_states()
    log.info("Validation Editdistance: %.4f" % (float(val_acc)))
    log.info("Time taken: %.2fs\n" % (time.time() - start_time))

def checkpoint_save(checkpoint, manager):
    checkpoint.step.assign_add(1)
    if int(checkpoint.step) % 1 == 0:
        save_path = manager.save()
        print("Saved checkpoint for ckpt step {}: {}".format(int(checkpoint.step), save_path))

def train(input_shape1, input_shape2, num_classes, learning_rate,data,
          batch_size, size, scale, EPOCHS, SAVE_PATH, monitor, load_model,
          restore, number_train, number_valid):


    if scale != '':
        train1Data,valid1Data = tfdata2(data[0], data[1],data[2],data[3], batch_size, size, scale)
    else:
        train1Data,valid1Data = tfdata1(data[0],data[1] , batch_size, size)


    SAVE_PATH = SAVE_PATH + '{}_{}/'.format(scale, size)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # purpose: save and restore models
    checkpoint_path = SAVE_PATH +"training_checkpoints/"

    model = rnn_att_model(input_shape1, input_shape2, num_classes)
    # optimizer = keras.optimizers.Adam(learning_rate)
    optimizer = keras.optimizers.RMSprop(learning_rate)

    checkpoint = tf.train.Checkpoint(step = tf.Variable(1),
                                     model = model,
                                     generator_optimizer=optimizer)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print('check point not exist {}'.format(checkpoint_path))
    else:
        print('check point {}'.format(checkpoint_path))
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)
    if restore:
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")


    train_loss_results = []
    train_accuracy_results = []

    log = init_logging(SAVE_PATH)
    for epoch in range(EPOCHS):
        print('number of train samples {}'.format(number_train))
        print('number of valid samples {}'.format(number_valid))
        start_time = time.time()

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = EditDistance()
        val_accuracy = EditDistance()
        train_step(epoch, train1Data,model, optimizer,epoch_loss_avg, epoch_accuracy, log)

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        log.info("Epoch {:03d}: Loss: {:.3f}, Editdistance: {:.3}".format(epoch,epoch_loss_avg.result(),
                                                                         epoch_accuracy.result()))
        epoch_loss_avg.reset_states()
        epoch_accuracy.reset_states()
        checkpoint_save(checkpoint, manager)
        validation(valid1Data, model, val_accuracy, log, start_time)
        test(batch_size, model, log)








    # if load_model:
    #     model = keras.models.load_model(SAVE_PATH,  custom_objects={'tf': tf,
    #                                                                 'CTCLoss': CTCLoss, 'EditDistance':EditDistance})
    #     model.compile(optimizer=keras.optimizers.Adam(learning_rate),
    #                   loss=CTCLoss(), metrics=[EditDistance()])
    #     model.summary()
    # else:
    #     model = rnn_att_model(input_shape1, input_shape2, num_classes)
    #     model.compile(optimizer=keras.optimizers.Adam(learning_rate),
    #                   loss=CTCLoss(), metrics=[EditDistance()])
    #     model.summary()
    #


    # earlyStopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=3, verbose=1)
    # # Reduce LR on Plateau
    # reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=2, verbose=1)
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=SAVE_PATH,
    #     save_weights_only=False,
    #     monitor = monitor,
    #     mode='min',
    #     save_freq = int(train1steps*3),
    #     save_best_only=True)
    #
    # history_lstm = model.fit(train1Data.repeat(),
    #                          steps_per_epoch=train1steps,
    #                          validation_data=valid1Data.repeat(),
    #                          validation_steps=valid1steps,
    #                          epochs=EPOCHS,
    #                          callbacks=[earlyStopping, reduceLR, model_checkpoint_callback])
    #
    # ## Save model
    #
    # model.save(SAVE_PATH)
    #
    # # serialize weights to HDF5
    # model.save_weights(SAVE_PATH + "weights")
    # print("Saved lstm model to res")
    #
    # ## Save history data
    # with open(SAVE_PATH + "train_results.pickle", "wb") as handle:
    #     pickle.dump(history_lstm.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print("Saved lstm training history to res")


    # with open('res/train_results.pickle', 'rb') as handle:
    #     history = pickle.load(handle)



def decode(SAVE_PATH, max_length, batch_size, scale, size):
    # decoder = decoders.CTCGreedyDecoder(table_path)
    Data, steps, data_test, y_true = getDataTest(max_length, batch_size)
    SAVE_PATH = SAVE_PATH + '{}_{}/'.format(scale, size)
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
    print(np.shape(y_pred))
    return y_true, decoded
    # np.save('{}outputs{}_{}.npy', outputs)
    # np.save('{}decoded{}_{}.npy', decoded)