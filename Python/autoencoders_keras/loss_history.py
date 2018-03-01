# Author: Hamaad Musharaf Shah.
# The following references were used.
# https://keras.io
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://stackoverflow.com/questions/42177658/how-to-switch-backend-with-keras-from-tensorflow-to-theano
# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
# http://scikit-learn.org/stable/
# Book: Ian Goodfellow, Yoshua Bengio and Aaron Courville, "Deep Learning" - http://www.deeplearningbook.org
# Book: Aurelien Geron, "Hands-On Machine Learning with Scikit-Learn & Tensorflow" - https://www.amazon.co.uk/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291

import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, 
                       logs={}):
       self.log = []
 
    def on_epoch_end(self, 
                     batch, 
                     logs={}):
       self.log.append(logs)