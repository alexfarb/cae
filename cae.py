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


class PreProcessingData(object):
    # Metódo Construtor
    def __init__(self, dataset):
        self.dataset = dataset
    
    # Função para a divisão do conjunto em treino e teste    
    def split_train_test(self, data_train_num, data_test_num, data_length):
        input_train_original = self.dataset[0:data_train_num,0:data_length] # Conjunto de Treino (Entrada)
        output_train_original = input_train_original # Conjunto de Treino (Saída)
        input_test_original = self.dataset[data_train_num:data_test_num,0:data_length] # Conjunto de Teste (Entrada)
        output_test_original = input_test_original # Conjunto de Teste (Saída)
        return input_train_original, output_train_original, input_test_original, output_test_original
        
    # função para a normalização dos dados
    def normalize_data(self, data, min, max):
        return (data-min)/(max-min) # Cálculo para a normalização dos dados

# Classe para o Autoenconder  
class Autoencoder(object):
    # Metódo Construtor
    def __init__(self, input_train, input_test, output_train, output_test, data_length, data_train_num):
        self.input_train = input_train
        self.input_test = input_test
        self.output_train = output_train
        self.output_test = output_test
        self.data_length = data_length
        self.data_train_num = data_train_num
    
    # CAE para a reconstrução do sinal    
    def convolutional_autoencoder_1d(self, data_length, kernel_size, epochs_num, optimizer_function, loss_function) 
        x_train = np.expand_dims(self.input_train, axis=2) # Redimensionamento da entrada (treino) para o CAE
        y_train = np.expand_dims(self.output_train, axis=2) # Redimensionamento da saída (treino) para o CAE
        x_test = np.expand_dims(self.input_test, axis=2) # Redimensionamento da entrada (teste) para o CAE
        y_test = np.expand_dims(self.output_test, axis=2)# Redimensionamento da saída (teste) para o CAE    
        # Camadas de Convolução e Maxpooling para o Decoder    
        input_signal = Input(shape=(data_length,1))
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
        autoencoder.compile(optimizer=optimizer_function, loss=loss_function)
        
        # Treinamento do Modelo
        autoencoder.fit(x_train,y_train,
                         epochs=epochs_num,)
        
        # Saída do Treinamento
        decoded_train = autoencoder.predict(x_train)
        # Saída do Treinamento redimensionada
        decoded_train_reshaped = (decoded_train.reshape(data_train_num, data_length))
        # Erro de Treinamento
        train_error = self.input_train - decoded_train_reshaped
        return decoded_train_reshaped, train_error

# Classe para os Plots
class Plots(object):
    def __init__(self, data_length):
        self.data_length = data_length
        
    def plot_reconstruct(self, sample_name, input_data, decoded_data, train_error, data_row):
        title = "Paciente %s" %(sample_name)
        plt.suptitle(title)
        plt.subplot(311)
        plt.plot(input_data[data_row,:])
        plt.title("Sinal Original")
        plt.xlim(0,data_length)
        plt.subplot(312)
        plt.plot(decoded_data[data_row,:])
        plt.title("Sinal Reconstruído")
        plt.xlim(0,data_length)
        plt.subplot(313)
        plt.plot(train_error[data_row,:])
        plt.title("Erro entre o Sinal Original e o Reconstruído")
        plt.xlim(0,data_length)
        plt.show()
        

dataset = np.loadtxt("C:\\repos\\cae\\data\\conv1d\\healthy_samples_std.csv",delimiter=",") # Carrega a Base de Dados
data_length = len(dataset.T) # Tamanho do sinal em número de amostras
data_train_num = 14 # Tamanho do conjunto de treinamento
data_test_num = 18
kernel_size =  20
epochs_num = 2


