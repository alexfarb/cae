# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:58:32 2018

@author: Farb

Convolutional Auto-Encoder (CAE)

MIT-BIH Arrhythmia Database: https://www.physionet.org/physiobank/database/mitdb/

Converted to csv

"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K

class ReadFile(object): # Classe para a leitura da Base de Dados no Formato .csv

    def __init__(self, csv_size, total_rows):
        self.csv_size = csv_size
        self.total_rows = total_rows

    def read_single_row(self, file):
        with open(file, 'r') as arch:
            reader = csv.reader(arch, delimiter=',')
            file_data = []
            for p in range(0, self.csv_size):
                file_data.append(0)
            for row in reader:
                var_index = 0
                for value in row:
                    file_data[var_index] = float(value)
                    var_index += 1
        return file_data

    def read_multiple_rows(self, file):
        with open(file, 'r') as arch:
            index_row = 0
            reader = csv.reader(arch, delimiter=',')
            file_data = np.zeros((self.total_rows, self.csv_size))
            for row in reader:
                var_index = 0
                if row != []:
                    for value in row:
                        file_data[index_row][var_index] = float(value)
                        var_index += 1
                    index_row +=1
        return file_data
    
# Leitura da base de dados
csv_size = 3600 # Total de Amostras da Base de Dados
total_rows = 1 # Total de Linhas do arquivo .csv
data_train = 100 # Número da amostra que deseja ler
num_samples = 3600 # Número de amostras do paciente
input_train_data = "C:\\repos\\cae\\data\\diag\\normalized\\all\\%sm.csv" %(data_train)
target_train_data = "C:\\repos\\cae\\data\\diag\\standard\\all\\%sm.csv" %(data_train)

read_database = ReadFile(csv_size, total_rows) # Instanciamento para a leitura da Base de Dados

# Conjunto de Treino
input_train = np.array(read_database.read_single_row(input_train_data)) # Entrada Normalizada
input_train = input_train[0:num_samples]
input_train = input_train.reshape(num_samples,1)
target_train = np.array(read_database.read_single_row(target_train_data)) # Saída Desejada
target_train = target_train[0:num_samples]
target_train = target_train.reshape(num_samples,1)
input_train_std = target_train # Entrada sem a normalização

input_img = Input(shape=(num_samples,1))  # adapt this if using `channels_first` image data format

x = Conv1D(16, 12, padding='same')(input_img)
x = MaxPooling1D(4, padding='same')(x)
x = Conv1D(8, 12, padding='same')(x)
x = MaxPooling1D(4, padding='same')(x)
x = Conv1D(8, 12, padding='same')(x)
encoded = MaxPooling1D(4, padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv1D(8, 12, padding='same')(encoded)
x = UpSampling1D(4)(x)
x = Conv1D(8, 12, padding='same')(x)
x = UpSampling1D(4)(x)
x = Conv1D(16, 12)(x)
x = UpSampling1D(4)(x)
decoded = Conv1D(1, 12, activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

plt.plot(input_train)
plt.show()