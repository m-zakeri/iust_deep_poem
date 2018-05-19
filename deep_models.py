from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
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
    model = Sequential()
    model.add(LSTM(128, input_shape=input_dim, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(128, input_shape=input_dim, return_sequences=False))
    # model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_7'


#
#
#
def model_8(input_dim, output_dim):
    model = Sequential()
    model.add(LSTM(256, input_shape=input_dim, return_sequences=True, recurrent_dropout=0.1))
    model.add(Dropout(0.3))
    model.add(LSTM(256, input_shape=input_dim, return_sequences=False, recurrent_dropout=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_8'