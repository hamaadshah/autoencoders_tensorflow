# Author: Hamaad Musharaf Shah.

import math
import inspect

from sklearn.base import BaseEstimator, TransformerMixin

import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda, add
from keras.models import Model, Sequential

import tensorflow

from autoencoders_keras.loss_history import LossHistory

class VariationalAutoencoder(BaseEstimator, 
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
        
        self.mu = Dense(units=self.encoding_dim, activation="linear")(self.encoded)
        self.log_sigma = Dense(units=self.encoding_dim, activation="linear")(self.encoded)
        z = Lambda(self.sample_z, output_shape=(self.encoding_dim,))([self.mu, self.log_sigma])

        self.decoded_layers_dict = {}
        
        decoder_counter = 0
        
        for i in range(self.decoder_layers):
            if i == 0:
                self.decoded_layers_dict[decoder_counter] = BatchNormalization()
                decoder_counter += 1
                self.decoded_layers_dict[decoder_counter] = Dense(units=self.n_hidden_units, activation="elu")
                decoder_counter += 1
                self.decoded_layers_dict[decoder_counter] = Dropout(rate=0.5)

                self.decoded = self.decoded_layers_dict[decoder_counter - 2](z)
                self.decoded = self.decoded_layers_dict[decoder_counter - 1](self.decoded)
                self.decoded = self.decoded_layers_dict[decoder_counter](self.decoded)

                decoder_counter += 1
            elif i > 0 and i < self.decoder_layers - 1:
                self.decoded_layers_dict[decoder_counter] = BatchNormalization()
                decoder_counter += 1
                self.decoded_layers_dict[decoder_counter] = Dense(units=self.n_hidden_units, activation="elu")
                decoder_counter += 1
                self.decoded_layers_dict[decoder_counter] = Dropout(rate=0.5)

                self.decoded = self.decoded_layers_dict[decoder_counter - 2](self.decoded)
                self.decoded = self.decoded_layers_dict[decoder_counter - 1](self.decoded)
                self.decoded = self.decoded_layers_dict[decoder_counter](self.decoded)

                decoder_counter += 1
            elif i == self.decoder_layers - 1:
                self.decoded_layers_dict[decoder_counter] = BatchNormalization()
                decoder_counter += 1
                self.decoded_layers_dict[decoder_counter] = Dense(units=self.n_hidden_units, activation="elu")

                self.decoded = self.decoded_layers_dict[decoder_counter - 1](self.decoded)
                self.decoded = self.decoded_layers_dict[decoder_counter](self.decoded)
                decoder_counter += 1
        
        # Output would have shape: (batch_size, n_feat).
        self.decoded_layers_dict[decoder_counter] = Dense(units=self.n_feat, activation="sigmoid")
        self.decoded = self.decoded_layers_dict[decoder_counter](self.decoded)

        self.autoencoder = Model(self.input_data, self.decoded)
        self.autoencoder.compile(optimizer=keras.optimizers.Adam(),
                                 loss=self.vae_loss)
            
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

        self.encoder = Model(self.input_data, self.mu)

        self.generator_input = Input(shape=(self.encoding_dim,))
        self.generator_output = None
        decoder_counter = 0
            
        for i in range(self.decoder_layers):
            if i == 0:
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_input)
                decoder_counter += 1
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1
            elif i > 0 and i < self.decoder_layers - 1:
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1
            elif i == self.decoder_layers - 1:
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1

        self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)

        self.generator = Model(self.generator_input, self.generator_output)
                
        return self
    
    def transform(self,
                  X):
        return self.encoder.predict(X)
    
    def sample_z(self,
                 args):
        mu_, log_sigma_ = args
        eps = keras.backend.random_normal(shape=(keras.backend.shape(mu_)[0], self.encoding_dim),
                                          mean=0.0,
                                          stddev=1.0)
        out = mu_ + keras.backend.exp(log_sigma_ / 2) * eps
            
        return out
    
    def vae_loss(self,
                 y_true,
                 y_pred):
        recon = self.n_feat * keras.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
        kl = -0.5 * keras.backend.mean(1.0 + self.log_sigma - keras.backend.exp(self.log_sigma) - keras.backend.square(self.mu), axis=-1)
        out = keras.backend.mean(recon + kl)
            
        return out