import os
from PIL import Image
import json
import numpy as np
import pickle
from pprint import pprint

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Embedding

###
# Constants
###

MAX_HEADLINE_WORDS = 12

LETTERS = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM 1234567890.-|/'
EMBEDDING_DIM = len(LETTERS)+1
INPUT_LEN = 20 + 24

VALIDATION_SPLIT = 0.2


class Model(object):
    def __init__(self, outdir='model'):
        self._labels_index = {}
        self._outdir = outdir

        if not os.path.isdir(self._outdir):
            # 1st time
            os.mkdir(self._outdir)
            self._init_model()
        else:
            # Load model
            self._model = load_model(os.path.join(self._outdir, 'model.hdf5'))

    def train(self, epochs):
        x, y = self._load_training_data('data/small.jl')
        x_val, x_train, y_val, y_train = self._process_training_data(
            x, y)
        self._model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=epochs,
            validation_data=(x_val, y_val))
        self._save()

    def _save(self):
        self._model.save(os.path.join(self._outdir, 'model.hdf5'))


    def _prepare_input(self, domain, title):
        string = '{:20.20}{:24.24}'.format(domain, title)
        return [LETTERS.index(c) if c in LETTERS else len(LETTERS) for c in string]

    def _load_training_data(self, fp):
        X = []
        Y = []
        with open(fp) as f:
            for line in f:
                domain, title, y = json.loads(line)
                Y.append(y)
                X.append(self._prepare_input(domain, title))
        return X, Y

    def _process_training_data(self, x_raw, y_raw):
        # x and y as if we have the function that y = f(x)
        # where y is result, x is input
        x_all = np.array(x_raw)
        y_all = np.array(y_raw, 'float32') / 255

        # Split into training and validation
        rshuffle = np.arange(x_all.shape[0])  # np.arange ~= a (python) range
        np.random.shuffle(rshuffle)
        x_all = x_all[rshuffle]
        y_all = y_all[rshuffle]

        split_point = int(VALIDATION_SPLIT * x_all.shape[0])
        x_val = x_all[:split_point]
        x_train = x_all[split_point:]
        assert x_val.shape[0] + x_train.shape[0] == x_all.shape[0]
        y_val = y_all[:split_point]
        y_train = y_all[split_point:]
        assert y_val.shape[0] + y_train.shape[0] == y_all.shape[0]

        return x_val, x_train, y_val, y_train

    def _init_model(self):
        self._model = Sequential([
            # Input is implied because we are using the Sequential API
            # Input(shape=(MAX_HEADLINE_WORDS,), dtype='int32'),
            Embedding(len(LETTERS)+1,
                      EMBEDDING_DIM,
                      input_length=INPUT_LEN),
            Flatten(),
            Dense(256),
            Dense(128),
            Dense(256),
            Dense((16*16)*3, activation='softmax'),
        ])
        self._model.summary()
        self._model.compile(
            loss='mean_squared_error',
            optimizer='sgd')

    def predict(self, domain, title):
        x = self._prepare_input(domain, title)
        pred = self._model.predict(np.array([x]))[0]
        return pred


model = Model()
epochs = int(input('How many epochs to train for?  ') or '0')
if epochs:
    model.train(epochs)

while True:
    domain = input('Domain:  ')
    title = input('Title:  ')

    ret = model.predict(domain, title)
    pprint(ret)
    ret = np.array(ret)
    ret *= 255
    ret = ret.reshape((16, 16, 3))
    im = Image.fromarray(ret, mode='RGB')
    im.show()
