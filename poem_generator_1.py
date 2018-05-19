"""
Poem generator 1
-

"""

from __future__ import print_function

__version__ = '0.1.1'
__author__ = 'Morteza Zakeri'

import sys
import os
import datetime
import random
import numpy as np

from keras.models import load_model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, LambdaCallback
from keras.utils import plot_model

import poem_preprocess as preprocess
from config import dataset_config
import deep_models


class FileFormatFuzzer(object):
    """
    Main class for learn and fuzz process
    """
    def __init__(self, maxlen=40, step=1, batch_size=128):
        """

        :param maxlen:
        :param step:
        :param batch_size:
        """
        # learning hyper-parameters
        self.maxlen = maxlen
        self.step = step
        self.batch_size = batch_size

        self.text_all = ''
        self.text_training = ''
        self.text_validation = ''
        self.text_test = ''

        self.chars = None
        self.char_indices = None
        self.indices_char = None

        # self.model = None

        self.load_dataset()

    def define_model(self, input_dim, output_dim):
        """build the model: a single LSTM layer # we need to deep it"""
        model, model_name = deep_models.model_7(input_dim, output_dim)
        return model, model_name

    def load_dataset(self):
        """ load all 3 part of each dataset and building dictionary index """
        if dataset_config['dataset_size'] == 'small':
            self.text_training = preprocess.load_from_file(dataset_config['small_training_set_path'])
            self.text_validation = preprocess.load_from_file(dataset_config['small_validation_set_path'])
            self.text_test = preprocess.load_from_file(dataset_config['small_testing_set_path'])
        elif dataset_config['dataset_size'] == 'medium':
            self.text_training = preprocess.load_from_file(dataset_config['medium_training_set_path'])
            self.text_validation = preprocess.load_from_file(dataset_config['medium_validation_set_path'])
            self.text_test = preprocess.load_from_file(dataset_config['medium_testing_set_path'])
        elif dataset_config['dataset_size'] == 'large':
            self.text_training = preprocess.load_from_file(dataset_config['large_training_set_path'])
            self.text_validation = preprocess.load_from_file(dataset_config['large_validation_set_path'])
            self.text_test = preprocess.load_from_file(dataset_config['large_testing_set_path'])
        self.text_all = self.text_training + self.text_validation + self.text_test
        print('Total corpus length:', len(self.text_all))
        self.chars = sorted(list(set(self.text_all)))
        print('Total corpus chars:', len(self.chars))
        # print(chars)

        # Building dictionary index
        print('Building dictionary index ...')
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        # print(char_indices)
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        # print(indices_char)

    def generate_samples(self, text):
        """Cut the text in semi-redundant sequences of maxlen characters"""
        sentences = []  # List of all sentence as input
        next_chars = []  # List of all next chars as labels
        for i in range(0, len(text) - self.maxlen, self.step):  # arg2 why this?
            sentences.append(text[i: i + self.maxlen])
            # print(sentences)
            next_chars.append(text[i + self.maxlen])
            # print(next_chars)
        print('Number of semi sequences or samples:', len(sentences))
        return sentences, next_chars

    def data_generator(self, sentences, next_chars):
        """
        Batch data generator for large dataset not fit completely in memory
        # Index j now increase Shuffle

        :param sentences:
        :param next_chars:
        :return:
        """
        j = random.randint(0, len(sentences) - (self.batch_size+1))
        # print('Vectorization...')
        while True:
            # Fix generator :))
            x = np.zeros((self.batch_size, self.maxlen, len(self.chars)), dtype=np.bool)
            y = np.zeros((self.batch_size, len(self.chars)), dtype=np.bool)
            # j = random.randint(0, len(sentences) - (self.batch_size + 1))
            next_chars2 = next_chars[j: j + self.batch_size]  ## F...:)
            for i, one_sample in enumerate(sentences[j: j + self.batch_size]):
                for t, char in enumerate(one_sample):
                    x[i, t, self.char_indices[char]] = 1
                y[i, self.char_indices[next_chars2[i]]] = 1

            yield (x, y)
            # yield self.generate_single_batch(sentences, next_chars)
            j += self.batch_size
            if j > (len(sentences) - (self.batch_size+1)):
                j = random.randint(0, len(sentences) - (self.batch_size+1))

    def data_generator_validation(self, sentences, next_chars):
        """
                Batch data generator for large dataset not fit completely in memory
                # Index j now increase sequentially (validation don't need to shuffle)

                :param sentences:
                :param next_chars:
                :return:
                """
        j = 0
        # print('Vectorization...')
        while True:
            # Fix generator :))
            x = np.zeros((self.batch_size, self.maxlen, len(self.chars)), dtype=np.bool)
            y = np.zeros((self.batch_size, len(self.chars)), dtype=np.bool)
            # j = random.randint(0, len(sentences) - (self.batch_size + 1))
            next_chars2 = next_chars[j: j + self.batch_size]  ## F...:)
            for i, one_sample in enumerate(sentences[j: j + self.batch_size]):
                for t, char in enumerate(one_sample):
                    x[i, t, self.char_indices[char]] = 1
                y[i, self.char_indices[next_chars2[i]]] = 1

            yield (x, y)
            # yield self.generate_single_batch(sentences, next_chars)
            j += self.batch_size
            if j > (len(sentences) - (self.batch_size + 1)):
                j = 0

    def data_generator_in_memory(self, sentences, next_chars):
        """All data generate for small dataset fit completely in memory"""
        x = np.zeros((len(sentences), self.maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
        for i, one_sample in enumerate(sentences):
            for t, char in enumerate(one_sample):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1
        return x, y

    def train(self,
              epochs=1):
        """
        Create and train deep model

        :param epochs: Specify number of epoch for training.
        :return: Nothing.
        """
        # Main start time of training
        dt = datetime.datetime.now().strftime('_date_%Y-%m-%d_%H-%M-%S_')

        print('Generate training samples ...')
        sentences_training, next_chars_training = self.generate_samples(self.text_training)
        print('Generate validations samples ...')
        sentences_validation, next_chars_validation = self.generate_samples(self.text_validation)

        print('Build and compile model ...')
        model, model_name = self.define_model((self.maxlen, len(self.chars)), len(self.chars))
        optimizer = RMSprop(lr=0.01)  # [0.001, 0.01, 0.02, 0.05, 0.1]
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print(model_name, ' summary ...')
        model.summary()

        print(model_name, ' count_params ...')
        print(model.count_params())
        # input()

        print('Set #5 callback ...')
        # callback #1 EarlyStopping
        # monitor= 'val_loss' or monitor='loss'?
        model_early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=5, verbose=1, mode='auto')

        # callback #2 ModelCheckpoint
        # Create a directory for each training process to keep model checkpoint in .h5 format
        dir_name = './model_checkpoint/ferdosi/' + model_name + dt + 'epochs_' + str(epochs) + '/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name + model_name + dt + 'epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'
        model_chekpoint = ModelCheckpoint(file_name, verbose=1)

        # callback #3 TensorBoard
        dir_name = './logs_tensorboard/ferdosi/' + model_name + dt + 'epochs_' + str(epochs) + '/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        model_tensorboard = TensorBoard(log_dir=dir_name, histogram_freq=0, batch_size=self.batch_size,
                                        write_graph=True, write_grads=False, write_images=True, embeddings_freq=0,
                                        embeddings_layer_names=None, embeddings_metadata=None)

        # callback #4 CSVLogger
        # Create a directory and an empty csv file within to save mode csv log.
        dir_name = './logs_csv/ferdosi/' + model_name + dt + 'epochs_' + str(epochs) + '/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name + model_name + dt + '_epochs_' + str(epochs) + '_step_' + str(self.step) + '.csv'
        open(file_name, mode='a', newline='').close()
        model_csv_logger = CSVLogger(file_name, separator=',', append=False)

        # callback #5 LambdaCallback
        dir_name = './generated_results/ferdosi/' + model_name + dt + 'epochs_' + str(epochs) + '/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        def on_epoch_end(epoch, logs):
            nonlocal model
            nonlocal epochs
            nonlocal model_name
            nonlocal dir_name
            print('Sampling model and save results ... ')
            self.generate_and_fuzz_new_samples(model=model,
                                               model_name=model_name,
                                               epochs=epochs,
                                               current_epoch=epoch,
                                               dir_name=dir_name
                                               )

        generate_and_fuzz_new_samples_callback = LambdaCallback(on_epoch_begin=None,
                                                                on_epoch_end=on_epoch_end,
                                                                on_batch_begin=None,
                                                                on_batch_end=None,
                                                                on_train_begin=None,
                                                                on_train_end=None
                                                                )

        if dataset_config['dataset_size'] == 'small':  # very_small
            print('Start training on small dataset ...')
            x, y = self.data_generator_in_memory(sentences_training, next_chars_training)
            model.fit(x, y,
                      batch_size=self.batch_size,
                      epochs=epochs,
                      validation_split=0.2,
                      shuffle=True,
                      callbacks=[model_chekpoint,
                                 model_tensorboard,
                                 model_csv_logger,
                                 generate_and_fuzz_new_samples_callback]
                      )
        else:
            print('Start training on large dataset ...')
            print('Build training and validation data generators ...')
            training_data_generator = self.data_generator(sentences_training, next_chars_training)
            validation_data_generator = self.data_generator_validation(sentences_validation, next_chars_validation)
            model.fit_generator(generator=training_data_generator,
                                # steps_per_epoch=200,
                                steps_per_epoch=len(sentences_training) // self.batch_size,  # 1000,
                                validation_data=validation_data_generator,
                                validation_steps=len(sentences_validation) // (self.batch_size*2),  # 100,
                                # validation_steps=10,
                                use_multiprocessing=False,
                                workers=1,
                                epochs=epochs,
                                shuffle=True,
                                callbacks=[model_chekpoint,
                                           model_tensorboard,
                                           model_csv_logger,
                                           generate_and_fuzz_new_samples_callback]
                                )

    # end of train method
    # --------------------------------------------------------------------

    def generate_and_fuzz_new_samples(self,
                                      model=None,
                                      model_name='model_1',
                                      epochs=1,
                                      current_epoch=1,
                                      dir_name=None):
        """
        sampling the model and generate new object
        :param model: The model which is training.
        :param model_name: Name of model (base on hyperparameters config in deep_model.py file) e.g. [model_1, model_2,
        ...]
        :param epochs: Number of total epochs of training, e.g. 10,20,30,40,50 or 60
        :param current_epoch: Number of current epoch
        :param dir_name: root directory for this running.
        :return: Nothing
        """

        # End time of current epoch
        dt = datetime.datetime.now().strftime('_date_%Y-%m-%d_%H-%M-%S')
        dir_name = dir_name + 'epoch_' + str(current_epoch) + dt + '/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # diversities = [i*0.10 for i in range(1,20,2)]
        diversities = [0.2, 0.5, 1.0, 1.2, 1.5, 1.8]
        start_index = random.randint(0, len(self.test_text) - self.maxlen - 1)
        for diversity in diversities:
            print('-- diversity:', diversity)
            generated = ''
            sentence = self.text_test[start_index: start_index + self.maxlen]
            generated += sentence

            for i in range(1000):
                x_pred = np.zeros((1, self.maxlen, len(self.chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index, _, _, = self.sample(preds, diversity)
                next_char = self.indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

            # save generated_result to file inside program
            file_name = model_name \
                        + '_diversity_' + repr(diversity) \
                        + '_epochs_' + repr(epochs) \
                        + '_step_' + repr(self.step) \
                        + '.txt'
            preprocess.save_to_file(dir_name + file_name, generated)
            print('Diversity %s save to file successfully.' % diversity)

        print('End of generation method.')
        print('Starting new epoch ...')

    # Lower temperature will cause the model to make more likely,
    # but also more boring and conservative predictions.
    def sample(self, preds, temperature=1.0):
        """
        Helper function to sample an index from a probability array
        :param preds:
        :param temperature:
        :return:
        """

        # print('raw predictions = ', preds)
        preds = np.asarray(preds).astype('float64')

        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        # Sampling with numpy functions:
        probas = np.random.multinomial(1, preds, 1)
        # print()
        # print('sanitize predictions = ', preds)
        return np.argmax(probas), probas, preds

    def no_sample(self):
        pass

    def sample_space(self):
        pass

    def save_model_plot(self, model, epochs):
        """
        Save the model architecture plot.
        :param model:
        :param epochs:
        :return:
        """
        dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S_')
        # plot the model
        plot_model(model, to_file='./modelpic/date_' + dt + 'epochs_' + str(epochs) + '.png',
                   show_shapes=True, show_layer_names=True)

    def load_model_and_generate(self, model_name='model_1', epochs=1):
        dt = datetime.datetime.now().strftime('_date_%Y-%m-%d_%H-%M-%S')
        dir_name = './generated_results/ferdosi/' + model_name + dt + 'epochs_' + str(epochs) + '/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        model = load_model('./model_checkpoint/best_models/model_6_date_2018-05-05_11-16-05_epoch_17_val_loss_10.8927.h5')
        self.generate_and_fuzz_new_samples(model=model,
                                      model_name=model_name,
                                      epochs=1,
                                      current_epoch=1,
                                      dir_name=dir_name)

    def get_model_summary(self):
        print('Get model summary ...')
        model, model_name = self.define_model((self.maxlen, len(self.chars)), len(self.chars))
        print(model_name, ' summary ...')
        model.summary()
        print(model_name, ' count_params ...')
        print(model.count_params())


def main(argv):
    """ The main function to call train() method"""
    epochs = 100
    fff = FileFormatFuzzer(maxlen=30, step=1, batch_size=256)
    fff.train(epochs=epochs)
    # fff.get_model_summary()
    # fff.load_model_and_generate(model_name='model_6',
    #                             epochs=10)

    # previous_model_dir = './model_checkpoint/best_models/'
    # previous_model_name = 'date_20180325_200701_epoch_02_7.3107.h5'
    # previous_model_path = previous_model_dir + previous_model_name
    # model = load_model(previous_model_path)
    # fff.generate_and_fuzz_new_samples(model=model, model_name='best_models', maxlen=85, len_chars=96)
    print('training complete successfully on %s epochs' % epochs)


if __name__ == "__main__":
    main(sys.argv)
