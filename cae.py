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
    def __init__(self, input_train, input_test, output_train, output_test, data_length, data_train_num, data_test_num, data_dimension):
        self.input_train = input_train
        self.input_test = input_test
        self.output_train = output_train
        self.output_test = output_test
        self.data_length = data_length
        self.data_train_num = data_train_num
        self.data_test_num = data_test_num
        self.data_dimension = data_dimension
    
    # CAE para a reconstrução do sinal    
    def convolutional_autoencoder_1d(self, kernel_size, epochs_num, optimizer_option, loss_option):
        x_train = np.expand_dims(self.input_train, axis=2) # Redimensionamento da entrada (treino) para o CAE
        y_train = np.expand_dims(self.output_train, axis=2) # Redimensionamento da saída (treino) para o CAE
        x_test = np.expand_dims(self.input_test, axis=2) # Redimensionamento da entrada (teste) para o CAE
        y_test = np.expand_dims(self.output_test, axis=2)# Redimensionamento da saída (teste) para o CAE    
        # Camadas de Convolução e Maxpooling para o Decoder    
        input_signal = Input(shape=(self.data_length,self.data_dimension))
        x = Conv1D(self.data_dimension, kernel_size, padding='same')(input_signal)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(self.data_dimension, kernel_size, padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(self.data_dimension, kernel_size, padding='same')(x)
        encoded = MaxPooling1D(2, padding='same')(x) # Saída do Encoder
        
        # Camadas de Convolução e Upsampling para o Decoder
        x = Conv1D(self.data_dimension, kernel_size, padding='same')(encoded)
        x = UpSampling1D(2)(x)
        x = Conv1D(self.data_dimension, kernel_size, padding='same')(x)
        x = UpSampling1D(2)(x)
        x = Conv1D(self.data_dimension, kernel_size, padding='same')(x)
        x = UpSampling1D(2)(x)
        decoded = Conv1D(self.data_dimension, kernel_size, padding='same')(x) # Saída do Decoder
        
        # Compilação do Modelo
        autoencoder = Model(input_signal, decoded)
        autoencoder.compile(optimizer=optimizer_option, loss=loss_option)
        
        # Treinamento do Modelo
        history_callback = autoencoder.fit(x_train,y_train,
                         epochs=epochs_num)
        
        loss_history = history_callback.history["loss"]
        
        # Saída do Treinamento
        decoded_train = autoencoder.predict(x_train)
        # Saída do Treinamento redimensionada
        decoded_train_reshaped = (decoded_train.reshape(self.data_train_num, self.data_length))
        # Erro de Treinamento
        train_error = self.input_train - decoded_train_reshaped
        return decoded_train_reshaped, train_error, loss_history

# Classe para os Plots
class Plots(object):
    def __init__(self, input_data, decoded_data):
        self.input_data = input_data
        self.decoded_data = decoded_data
        
    def plot_reconstruct(self, sample_name, train_error, data_row, data_length):
        title = "Paciente %s" %(sample_name)
        plt.suptitle(title)
        plt.subplot(311)
        plt.plot(self.input_data[data_row,:])
        plt.title("Sinal Original")
        plt.xlim(0,data_length)
        plt.subplot(312)
        plt.plot(self.decoded_data[data_row,:])
        plt.title("Sinal Reconstruído")
        plt.xlim(0,data_length)
        plt.subplot(313)
        plt.plot(train_error[data_row,:])
        plt.title("Erro entre o Sinal Original e o Reconstruído")
        plt.xlim(0,data_length)
        plt.show()

network_num = 2

if network_num == 1:
    # Pré-processamento e parâmetros para a Base de Dados Saudável (Rede A)
    dataset_a = np.loadtxt("C:\\repos\\cae\\data\\conv1d\\healthy_samples_std.csv",delimiter=",") # Carrega a Base de Dados
    data_length_a = len(dataset_a.T) # Tamanho do sinal em número de amostras
    data_train_num_a = 14 # Tamanho do conjunto de treinamento
    data_test_num_a = 18 # Última amostra do conjunto de teste
    kernel_size_a =  20 # Tamanho do Kernel (Janela) de Convolução
    epochs_num_a = 2 # Quantidade de Épocas para o Treinamento da Rede
    data_dimension_a = 1 # Dimensão dos Dados
    optimizer_a = 'adamax'
    loss_a = 'mean_squared_error'
    # Divisão em treino e teste
    pre_processing_a = PreProcessingData(dataset_a)
    input_train_a, output_train_a, input_test_a, output_test_a = pre_processing_a.split_train_test(data_train_num_a, data_test_num_a, data_length_a)
    # Minimos e máximos do Conjunto de treino
    max_a = input_train_a.max()
    min_a = input_train_a.min()
    # Normalização das entradas e saídas
    x_train_a = pre_processing_a.normalize_data(input_train_a,min_a,max_a)
    y_train_a = x_train_a
    x_test_a = pre_processing_a.normalize_data(input_test_a,min_a,max_a)
    y_test_a = x_test_a

    # Autoencoder para Rede A
    auto_encoder_a = Autoencoder(x_train_a, x_test_a, y_train_a, y_test_a, data_length_a, data_train_num_a, data_test_num_a, data_dimension_a)
    decoded_train_a, train_error_a, loss_a = auto_encoder_a.convolutional_autoencoder_1d(kernel_size_a, epochs_num_a, optimizer_a, loss_a)

    # # Plots para a Rede A
    # plot = Plots(x_train_a, decoded_train_a)
    # plot.plot_reconstruct(100, train_error_a, 0, data_length_a)

else:
    # Pré-processamento e parâmetros para a Base de Dados com anormalidade (Rede B)
    dataset_b = np.loadtxt("C:\\repos\\cae\\data\\conv1d\\abnormal_samples_std.csv",delimiter=",") # Carrega a Base de Dados
    data_length_b = len(dataset_b.T) # Tamanho do sinal em número de amostras
    data_train_num_b = 20 # Tamanho do conjunto de treinamento
    data_test_num_b = 28 # Última amostra do conjunto de teste
    kernel_size_b =  20 # Tamanho do Kernel (Janela) de Convolução
    epochs_num_b = 10000 # Quantidade de Épocas para o Treinamento da Rede
    data_dimension_b = 1 # Dimensão dos Dados
    optimizer_b = 'adamax'
    loss_b = 'mean_squared_error'
    # Divisão em treino e teste
    pre_processing_b = PreProcessingData(dataset_b)
    input_train_b, output_train_b, input_test_b, output_test_b = pre_processing_b.split_train_test(data_train_num_b, data_test_num_b, data_length_b)
    # Minimos e máximos do Conjunto de treino
    max_b = input_train_b.max()
    min_b = input_train_b.min()
    # Normalização das entradas e saídas
    x_train_b = pre_processing_b.normalize_data(input_train_b,min_b,max_b)
    y_train_b = x_train_b
    x_test_b = pre_processing_b.normalize_data(input_test_b,min_b,max_b)
    y_test_b = x_test_b

    # Autoencoder para Rede B
    auto_encoder_b = Autoencoder(x_train_b, x_test_b, y_train_b, y_test_b, data_length_b, data_train_num_b, data_test_num_b, data_dimension_b)
    decoded_train_b, train_error_b, loss_b = auto_encoder_b.convolutional_autoencoder_1d(kernel_size_b, epochs_num_b, optimizer_b, loss_b)

    # # Plots para a Rede B
    # plot = Plots(x_train_b, decoded_train_b)
    # plot.plot_reconstruct(107, train_error_b, 0, data_length_b)