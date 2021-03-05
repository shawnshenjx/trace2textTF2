import tensorflow as tf
from tensorflow import keras

class CTCLoss(keras.losses.Loss):
    def __init__(self, logits_time_major=False, blank_index=0,
                 reduction=keras.losses.Reduction.AUTO, name='ctc_loss'):
        super().__init__(reduction=reduction, name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        label_length = tf.fill([tf.shape(y_true)[0]], tf.shape(y_true)[1])
        # y_pred = self.to_dense(y_pred,[tf.shape(y_true)[0], tf.shape(y_true)[1]])
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index)
        return tf.reduce_mean(loss)
    # def to_dense(self, tensor, shape):
    #     tensor = tf.sparse.reset_shape(tensor, shape)
    #     tensor = tf.sparse.to_dense(tensor, default_value=0)
    #     tensor = tf.cast(tensor, tf.int32)
    #     return tensor

def rnn_att_model(input_shape1,
                input_shape2,
                  output_shape,
                  cnn_features=10,
                  rnn='LSTM',
                  multi_rnn=True,
                  attention=False,
                  dropout=0.2):
    '''
    Long-Short-Term-Memory model

    Parameters:\n
    input_shape (array): dimensions of the model input\n
    cnn_features (int): number of features for the first CNN Layer\n
    rnn (string [LSTM, GRU]): type of RNN to use in the model\n
    multi_rnn (bool): activate or deactivate the second RNN Layer\n
    attention (bool): activate or deactivate the Attention Layer\n
    dropout (int [0:1]): dropout level for Dense Layers

    Returns:\n
    tf.keras.Model: Model built with keras
    '''

    # Fetch input
    input_shape = (input_shape1, input_shape2)
    inputs = tf.keras.Input(shape=input_shape)
    # reshape = tf.keras.layers.Reshape(
    #     input_shape=input_shape, target_shape=(input_shape1, input_shape2, 1))(inputs)
    #
    # # Normalization Layer
    # layer_out = tf.keras.layers.BatchNormalization()(reshape)
    #
    # # Convolutional Layer
    # layer_out = tf.keras.layers.Conv2D(cnn_features, kernel_size=(3, 3),
    #                                    padding='same', activation='relu')(layer_out)
    # layer_out = tf.keras.layers.BatchNormalization()(layer_out)
    # layer_out = tf.keras.layers.Conv2D(1, kernel_size=(3, 3),
    #                                    padding='same', activation='relu')(layer_out)
    # layer_out = tf.keras.layers.BatchNormalization()(layer_out)
    # layer_out = tf.keras.layers.Lambda(
    #     lambda x: tf.keras.backend.squeeze(x, -1), name='squeeze_dim')(layer_out)

    # LSTM Layer
    if rnn not in ['LSTM', 'GRU']:
        raise ValueError(
            'rnn should be equal to LSTM or GRU. No model generated...')

    if rnn == 'LSTM':
        layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            400, return_sequences=True, dropout=dropout))(inputs)
        if multi_rnn:
            layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                400, return_sequences=True, dropout=dropout))(layer_out)

    # GRU Layer
    if rnn == 'GRU':
        layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            400, return_sequences=True, dropout=dropout))(inputs)
        if multi_rnn:
            layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                400, return_sequences=True, dropout=dropout))(inputs)

    # Attention Layer 0.38
    if attention:
        query, value = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=2, axis=2))(layer_out)
        layer_out = tf.keras.layers.Attention(name='Attention')([query, value])

    # Classification Layer
    outputs = tf.keras.layers.Dense(output_shape)(layer_out)

    # Output Model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model