# License
# Copyright 2018 Hamaad Musharaf Shah
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import math
import inspect

from sklearn.base import BaseEstimator, TransformerMixin

import keras
from keras.layers import Input, Activation, Dense, Dropout, CuDNNLSTM, RepeatVector, TimeDistributed
from keras.models import Model

import tensorflow

from autoencoders_keras.loss_history import LossHistory

class Seq2SeqAutoencoder(BaseEstimator, 
                         TransformerMixin):
    def __init__(self, 
                 input_shape=None,
                 n_epoch=None,
                 batch_size=None,
                 encoder_layers=None,
                 decoder_layers=None,
                 n_hidden_units=None,
                 encoding_dim=None,
                 stateful=None,
                 denoising=None):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        
        for arg, val in values.items():
            setattr(self, arg, val)
        
        loss_history = LossHistory()
        
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=10)
        
        reduce_learn_rate = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                              factor=0.1,
                                                              patience=20)
        
        self.callbacks_list = [loss_history, early_stop, reduce_learn_rate]
        
        # 2D-lattice with time on the x-axis (across rows) and with space on the y-axis (across columns).
        if self.stateful is True:
            self.input_data = Input(batch_shape=self.input_shape)
            self.n_rows = self.input_shape[1]
            self.n_cols = self.input_shape[2]
        else:
            self.input_data = Input(shape=self.input_shape)
            self.n_rows = self.input_shape[0]
            self.n_cols = self.input_shape[1]

        for i in range(self.encoder_layers):
            if i == 0:
                # Returns a sequence of n_rows vectors of dimension n_hidden_units.
                self.encoded = CuDNNLSTM(units=self.n_hidden_units, return_sequences=True, stateful=self.stateful)(self.input_data)
            else:
                self.encoded = CuDNNLSTM(units=self.n_hidden_units, return_sequences=True, stateful=self.stateful)(self.encoded)

        # Returns 1 vector of dimension encoding_dim.
        self.encoded = CuDNNLSTM(units=self.encoding_dim, return_sequences=False, stateful=self.stateful)(self.encoded)

        # Returns a sequence containing n_rows vectors where each vector is of dimension encoding_dim.
        # output_shape: (None, n_rows, encoding_dim).
        self.decoded = RepeatVector(self.n_rows)(self.encoded)

        for i in range(self.decoder_layers):
            self.decoded = CuDNNLSTM(units=self.n_hidden_units, return_sequences=True, stateful=self.stateful)(self.decoded)
        
        # If return_sequences is True: 3D tensor with shape (batch_size, timesteps, units).
        # Else: 2D tensor with shape (batch_size, units).
        # Note that n_rows here is timesteps and n_cols here is units.
        # If return_state is True: a list of tensors. 
        # The first tensor is the output. The remaining tensors are the last states, each with shape (batch_size, units).
        # If stateful is True: the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
        # For LSTM (not CuDNNLSTM) If unroll is True: the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.
        self.decoded = CuDNNLSTM(units=self.n_cols, return_sequences=True, stateful=self.stateful)(self.decoded)

        self.autoencoder = Model(self.input_data, self.decoded)
        self.autoencoder.compile(optimizer=keras.optimizers.Adam(),
                                 loss="mean_squared_error")
            
    def fit(self,
            X,
            y=None):
        self.autoencoder.fit(X if self.denoising is None else X + self.denoising, X,
                             validation_split=0.3,
                             epochs=self.n_epoch,
                             batch_size=self.batch_size,
                             shuffle=True,
                             callbacks=self.callbacks_list,
                             verbose=1)

        self.encoder = Model(self.input_data, self.encoded)
        
        return self
    
    def transform(self,
                  X):
        return self.encoder.predict(X)