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
import matplotlib.pyplot as plt
# import plotly as py
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model, Sequential
from keras import backend as K
#from keras.callbacks import TensorBoard
from sklearn import preprocessing

dataset = np.loadtxt("C:\\repos\\cae\\data\\conv1d\\healthy_samples_std.csv",delimiter=",") # Carrega a Base de Dados
data_length = len(dataset.T) # Tamanho do sinal em número de amostras
data_train_num = 9 # Tamanho do conjunto de treinamento
data_test_num = 18
data_dimension = 1 # Dimensionalidade dos Dados
kernel_size =  20
epochs_num = 2
input_train_original = dataset[0:data_train_num,0:data_length] # Conjunto de Treino (Entrada)
output_train_original = input_train_original # Conjunto de Treino (Saída)
input_test_original = dataset[data_train_num+1:data_test_num,0:data_length] # Conjunto de Teste (Entrada)
output_test_original = input_test_original # Conjunto de Teste (Saída)
# Normalização da entrada
min_max_scaler = preprocessing.MinMaxScaler()
input_train_minmax = min_max_scaler.fit_transform(input_train_original)

'''
x_train = np.expand_dims(input_train, axis=2) # Redimensionamento da entrada (treino) para o CAE
y_train = np.expand_dims(output_train, axis=2) # Redimensionamento da saída (treino) para o CAE
x_test = np.expand_dims(input_test, axis=2) # Redimensionamento da entrada (teste) para o CAE
y_test = np.expand_dims(output_test, axis=2)# Redimensionamento da saída (teste) para o CAE
'''
'''
# Camadas de Convolução e Pooling para o Encoder
input_signal = Input(shape=(data_length,data_dimension))
x = Conv1D(data_dimension, kernel_size, padding='same')(input_signal)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(data_dimension, kernel_size, padding='same')(x)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(data_dimension, kernel_size, padding='same')(x)
encoded = MaxPooling1D(2, padding='same')(x) # Saída do Encoder

# Camadas de Convolução e Upsampling para o Decoder
x = Conv1D(data_dimension, kernel_size, padding='same')(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(data_dimension, kernel_size, padding='same')(x)
x = UpSampling1D(2)(x)
x = Conv1D(data_dimension, kernel_size, padding='same')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(data_dimension, kernel_size, padding='same')(x) # Saída do Decoder

# Compilação do Modelo
autoencoder = Model(input_signal, decoded)
autoencoder.compile(optimizer='adamax', loss='mean_squared_error')

# Treinamento do Modelo
autoencoder.fit(x_train,y_train,
                 epochs=epochs_num,)

# Saída do Treinamento
decoded_train = autoencoder.predict(x_train)
# Saída do Treinamento redimensionada
decoded_train_reshaped = (decoded_train.reshape(data_train_num, data_length))
# Erro de Treinamento
train_error = input_train - decoded_train_reshaped

# Plots
plt.suptitle("Paciente 100m")
plt.subplot(311)
plt.plot(input_train[0,:])
plt.title("Sinal Original")
plt.xlim(0,data_length)
plt.subplot(312)
plt.plot(decoded_train_reshaped[0,:])
plt.title("Sinal Reconstruído")
plt.xlim(0,data_length)
plt.subplot(313)
plt.plot(train_error[0,:])
plt.title("Erro entre o Sinal Original e o Reconstruído")
plt.xlim(0,data_length)
plt.show()

'''
