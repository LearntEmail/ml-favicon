import os
from PIL import Image
import json
import numpy as np
import pickle
from pprint import pprint
from glob import glob
from tqdm import tqdm
import subprocess
from tempfile import mkstemp

import keras
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, LocallyConnected2D

###
# Constants
###

OUTPUT_LEN = (16*16*3)
INPUT_SHAPE = (320, 320, 3)

VALIDATION_SPLIT = 0.1

USER_IMAGE_CACHE = 'user-image-cache'

if not os.path.isdir(USER_IMAGE_CACHE):
    os.mkdir(USER_IMAGE_CACHE)


class Model(object):
    def __init__(self, outdir='model'):
        self._labels_index = {}
        self._outdir = outdir
        self._training_data = None

        if not os.path.isdir(self._outdir):
            self._init_model()
        else:
            # Load model
            self._model = load_model(os.path.join(self._outdir, 'model.hdf5'))

    def train(self, epochs):
        if self._training_data is None:
            if os.path.isfile('tdcache.npy.npz'):
                dic = np.load('tdcache.npy.npz')
                self._training_data = [dic[k] for k in sorted(dic.keys())]
            else:
                x, y = self._load_training_data('data/small.jl')
                self._training_data = self._process_training_data(x, y)
                np.savez('tdcache.npy', *self._training_data)

        x_val, x_train, y_val, y_train = self._training_data
        self._model.fit(
            x_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[
                keras.callbacks.TensorBoard(
                    histogram_freq=10),
                keras.callbacks.ReduceLROnPlateau(),
            ])

    def save(self):
        if not os.path.isdir(self._outdir):
            # 1st time
            os.mkdir(self._outdir)
        self._model.save(os.path.join(self._outdir, 'model.hdf5'))

    def _prepare_input(self, fp):
        xim = Image.open(fp).resize(INPUT_SHAPE[:2], Image.NORMAL).convert('RGB')
        return np.array(xim.getdata()).flatten().reshape(INPUT_SHAPE)

    def _load_training_data(self, fp):
        X = []
        Y = []
        for fp in tqdm(glob('data/*/X.png')):
            X.append(self._prepare_input(fp))

            yfp = fp.replace('X.png', 'Y.png')
            yim = Image.open(yfp).convert('RGB')
            y = np.array(yim.getdata()).flatten()
            Y.append(y)
        return X, Y

    def _process_training_data(self, x_raw, y_raw):
        # x and y as if we have the function that y = f(x)
        # where y is result, x is input
        x_all = np.array(x_raw, 'float16') / 255
        y_all = np.array(y_raw, 'float16') / 255

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
            Conv2D(32, 2, input_shape=INPUT_SHAPE),
            MaxPooling2D(3),
            Conv2D(32, 2),
            MaxPooling2D(3),
            Conv2D(32, 2),
            MaxPooling2D(3),
            Conv2D(32, 2),
            MaxPooling2D(3),
            Flatten(),
            Dense(OUTPUT_LEN),
        ])
        self._model.summary()
        self._model.compile(
            loss='mean_squared_error',
            optimizer='sgd')

    def predict(self, fp):
        x = self._prepare_input(fp)
        pred = self._model.predict(np.array([x], 'float16'))[0]
        return pred


model = Model()
print('Training model.  Hit Ctrl-C to save and begin inference')
try:
    while True:
        model.train(20)
except KeyboardInterrupt:
    print('Stopping training')
    model.save()


while True:
    domain = input('Domain:  ')
    url = 'https://{}/'.format(domain)
    fp = os.path.join(USER_IMAGE_CACHE, domain+'.png')
    
    if not os.path.isfile(fp):
        subprocess.call(['python3', 'thumbnail.py', url, fp])

    ret = model.predict(fp)
    ret = np.array(ret)
    ret *= 255
    ret = ret.reshape((16, 16, 3))
    ret = np.array(ret, np.uint8)
    im = Image.fromarray(ret, mode='RGB')
    # Otherwise the preview window is tiny
    im = im.resize((512, 512))
    im.show()
