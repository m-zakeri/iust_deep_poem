from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Bidirectional
from keras.layers import LSTM


# Keras LSTM text generation example model (simplest model)
# summery of result for model_0 (Not deep model):
#
#
def model_0(input_dim, output_dim):
    """
    Total params: 127,584
    Trainable params: 127,584
    Non-trainable params: 0

    :param input_dim:
    :param output_dim:
    :return:
    """
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=input_dim))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_0'


#
#
#
def model_7(input_dim, output_dim):
    """
    model_7  summary ...
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    lstm_1 (LSTM)                (None, 40, 128)           90624
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 40, 128)           0
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 128)               131584
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 48)                6192
    _________________________________________________________________
    activation_1 (Activation)    (None, 48)                0
    =================================================================
    Total params: 228,400
    Trainable params: 228,400
    Non-trainable params: 0
    _________________________________________________________________
    model_7  count_params ...
    228400
    :param input_dim:
    :param output_dim:
    :return:
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=input_dim, return_sequences=True, recurrent_dropout=0.1))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_7'


#
#
#
def model_8(input_dim, output_dim):
    """
    model_8  summary ...
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    lstm_1 (LSTM)                (None, 40, 64)            28928
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 64)                33024
    _________________________________________________________________
    dense_1 (Dense)              (None, 48)                3120
    _________________________________________________________________
    activation_1 (Activation)    (None, 48)                0
    =================================================================
    Total params: 65,072
    Trainable params: 65,072
    Non-trainable params: 0
    _________________________________________________________________
    model_8  count_params ...
    65072
    :param input_dim:
    :param output_dim:
    :return:
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=input_dim, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_8'


#
#
#
def model_9(input_dim, output_dim):
    """
    model_9  summary ...
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    bidirectional_1 (Bidirection (None, 40, 64)            57856
    _________________________________________________________________
    bidirectional_2 (Bidirection (None, 64)                66048
    _________________________________________________________________
    dense_1 (Dense)              (None, 48)                3120
    _________________________________________________________________
    activation_1 (Activation)    (None, 48)                0
    =================================================================
    Total params: 127,024
    Trainable params: 127,024
    Non-trainable params: 0
    _________________________________________________________________
    model_9  count_params ...
    127024

    :param input_dim:
    :param output_dim:
    :return:
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True),
                            input_shape=input_dim,
                            merge_mode='sum'))
    model.add(Bidirectional(LSTM(64, return_sequences=False),
                            merge_mode='sum'))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_9'