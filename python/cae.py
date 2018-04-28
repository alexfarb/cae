# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:58:32 2018

@author: Farb

Convolutional Auto-Encoder (CAE)

MIT-BIH Arrhythmia Database: https://www.physionet.org/physiobank/database/mitdb/

Converted to csv

"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import csv
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model, Sequential
from keras import backend as K

dataset = np.loadtxt("C:\\repos\\cae\\data\\conv1d\\healthy_samples_std.csv",delimiter=",")
x_train = np.expand_dims(dataset, axis=2)
y_train = np.expand_dims(dataset, axis=2)

input_img = Input(shape=(3600,1))  # adapt this if using `channels_first` image data format
print(input_img)
print("ENCODED")
x = Conv1D(1, 36, padding='same')(input_img)
print(x)
x = MaxPooling1D(5, padding='same')(x)
print(x)
x = Conv1D(1, 36, padding='same')(x)
print(x)
x = MaxPooling1D(5, padding='same')(x)
print(x)
x = Conv1D(1, 36, padding='same')(x)
print(x)
encoded = MaxPooling1D(4, padding='same')(x)
print(encoded)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
print("DECODED")
x = Conv1D(1, 36, padding='same')(encoded)
print(x)
x = UpSampling1D(4)(x)
print(x)
x = Conv1D(1, 36, padding='same')(x)
print(x)
x = UpSampling1D(5)(x)
print(x)
x = Conv1D(1, 36, padding='same')(x)
print(x)
x = UpSampling1D(5)(x)
print(x)
decoded = Conv1D(1, 36, activation='relu', padding='same')(x)
print(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train,y_train,
                 epochs=50,
                 shuffle=True)

# plt.plot(input_train)
# plt.show()

