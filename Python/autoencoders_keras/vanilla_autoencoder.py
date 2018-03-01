# Author: Hamaad Musharaf Shah.

import math
import inspect

from sklearn.base import BaseEstimator, TransformerMixin

import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model

import tensorflow

from autoencoders_keras.loss_history import LossHistory

class VanillaAutoencoder(BaseEstimator, 
                         TransformerMixin):
    def __init__(self, 
                 n_feat=None,
                 n_epoch=None,
                 batch_size=None,
                 encoder_layers=None,
                 decoder_layers=None,
                 n_hidden_units=None,
                 encoding_dim=None,
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
        
        for i in range(self.encoder_layers):
            if i == 0:
                self.input_data = Input(shape=(self.n_feat,))
                self.encoded = BatchNormalization()(self.input_data)
                self.encoded = Dense(units=self.n_hidden_units, activation="elu")(self.encoded)
                self.encoded = Dropout(rate=0.5)(self.encoded)
            elif i > 0 and i < self.encoder_layers - 1:
                self.encoded = BatchNormalization()(self.encoded)
                self.encoded = Dense(units=self.n_hidden_units, activation="elu")(self.encoded)
                self.encoded = Dropout(rate=0.5)(self.encoded)
            elif i == self.encoder_layers - 1:
                self.encoded = BatchNormalization()(self.encoded)
                self.encoded = Dense(units=self.n_hidden_units, activation="elu")(self.encoded)
        
        self.encoded = BatchNormalization()(self.encoded)
        self.encoded = Dense(units=self.encoding_dim, activation="sigmoid")(self.encoded)

        for i in range(self.decoder_layers):
            if i == 0:
                self.decoded = BatchNormalization()(self.encoded)
                self.decoded = Dense(units=self.n_hidden_units, activation="elu")(self.decoded)
                self.decoded = Dropout(rate=0.5)(self.decoded)
            elif i > 0 and i < self.decoder_layers - 1:
                self.decoded = BatchNormalization()(self.decoded)
                self.decoded = Dense(units=self.n_hidden_units, activation="elu")(self.decoded)
                self.decoded = Dropout(rate=0.5)(self.decoded)
            elif i == self.decoder_layers - 1:
                self.decoded = BatchNormalization()(self.decoded)
                self.decoded = Dense(units=self.n_hidden_units, activation="elu")(self.decoded)
        
        # Output would have shape: (batch_size, n_feat).
        self.decoded = BatchNormalization()(self.decoded)
        self.decoded = Dense(units=self.n_feat, activation="sigmoid")(self.decoded)

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