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
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K

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
    def convolutional_autoencoder_1d(self, kernel_size, epochs_num, optimizer_option, loss_option, cae_name):
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
                         epochs=epochs_num,
                         validation_data = (x_test, y_test))
        # Salva o histórico do erro para treino e teste
        loss_train = history_callback.history["loss"]
        loss_test = history_callback.history["val_loss"]

        # Salva o modelo treinado em arquivo .h5
        autoencoder.save(cae_name)
        
        # Saída do Treinamento
        decoded_train = autoencoder.predict(x_train)
        decoded_test = autoencoder.predict(x_test)
        # Saída do Treinamento redimensionada
        decoded_train_reshaped = (decoded_train.reshape(self.data_train_num, self.data_length))
        decoded_test_reshaped = (decoded_test.reshape(self.data_test_num-self.data_train_num, self.data_length))
        # Erro de Treinamento
        return decoded_train_reshaped, decoded_test_reshaped, loss_train, loss_test

def main_cae():
    option = [1, 2]
    epochs_ab = 3000 # 100, 500, 1000, 10000
    max_ab = 1526
    min_ab = 403


    for i in range(0,2):
        if option[i] == 1:
            # Pré-processamento e parâmetros para a Base de Dados Saudável (Rede A)
            dataset_a = np.loadtxt("C:\\repos\\cae\\data\\conv1d\\healthy_samples_std.csv",delimiter=",") # Carrega a Base de Dados
            data_length_a = len(dataset_a.T) # Tamanho do sinal em número de amostras
            data_train_num_a = 14 # Tamanho do conjunto de treinamento
            data_test_num_a = 18 # Última amostra do conjunto de teste
            kernel_size_a =  20 # Tamanho do Kernel (Janela) de Convolução
            epochs_num_a = epochs_ab # Quantidade de Épocas para o Treinamento da Rede
            data_dimension_a = 1 # Dimensão dos Dados
            optimizer_a = 'adamax'
            loss_a = 'mean_squared_error'
            cae_name_a = 'C:\\repos\\cae\saved_models\\cae_a.h5'
            # Divisão em treino e teste
            pre_processing_a = PreProcessingData(dataset_a)
            input_train_a, output_train_a, input_test_a, output_test_a = pre_processing_a.split_train_test(data_train_num_a, data_test_num_a, data_length_a)
            # Minimos e máximos do Conjunto de treino
        #    max_a = input_train_a.max()
            max_a = max_ab
        #    print(max_a)
        #    min_a = input_train_a.min()
            min_a = min_ab
        #    print(min_a)
            # Normalização das entradas e saídas
            x_train_a = pre_processing_a.normalize_data(input_train_a,min_a,max_a)
            y_train_a = x_train_a
            x_test_a = pre_processing_a.normalize_data(input_test_a,min_a,max_a)
            y_test_a = x_test_a
        
            # Autoencoder para Rede A
            auto_encoder_a = Autoencoder(x_train_a, x_test_a, y_train_a, y_test_a, data_length_a, data_train_num_a, data_test_num_a, data_dimension_a)
            decoded_train_a, decoded_test_a, loss_train_a, loss_test_a = auto_encoder_a.convolutional_autoencoder_1d(kernel_size_a, epochs_num_a, optimizer_a, loss_a, cae_name_a)
            decoded_train_a_original = decoded_train_a*(max_a-min_a)+min_a
            decoded_test_a_original = decoded_test_a*(max_a-min_a)+min_a
            train_error_a = input_train_a - decoded_train_a_original
            test_error_a = input_test_a - decoded_test_a_original
        
            # Salva os resultados em um arquivo .csv
            np.savetxt('C:\\repos\\cae\\results\\input_train_a.csv', input_train_a, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\input_test_a.csv', input_test_a, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\decoded_train_a.csv', decoded_train_a_original, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\decoded_test_a.csv', decoded_test_a_original, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\train_error_a.csv', train_error_a, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\test_error_a.csv', test_error_a, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\loss_train_a.csv', loss_train_a, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\loss_test_a.csv', loss_test_a, delimiter=',', fmt='%s')
        
        elif option[i] == 2:
            # Pré-processamento e parâmetros para a Base de Dados com anormalidade (Rede B)
            dataset_b = np.loadtxt("C:\\repos\\cae\\data\\conv1d\\abnormal_samples_std.csv",delimiter=",") # Carrega a Base de Dados
            data_length_b = len(dataset_b.T) # Tamanho do sinal em número de amostras
            data_train_num_b = 20 # Tamanho do conjunto de treinamento
            data_test_num_b = 28 # Última amostra do conjunto de teste
            kernel_size_b =  20 # Tamanho do Kernel (Janela) de Convolução
            epochs_num_b = epochs_ab # Quantidade de Épocas para o Treinamento da Rede
            data_dimension_b = 1 # Dimensão dos Dados
            optimizer_b = 'adamax'
            loss_b = 'mean_squared_error'
            cae_name_b = 'C:\\repos\\cae\saved_models\\cae_b.h5'
            # Divisão em treino e teste
            pre_processing_b = PreProcessingData(dataset_b)
            input_train_b, output_train_b, input_test_b, output_test_b = pre_processing_b.split_train_test(data_train_num_b, data_test_num_b, data_length_b)
            # Minimos e máximos do Conjunto de treino
        #    max_b = input_train_b.max()
        #    min_b = input_train_b.min()
            max_b = max_ab
            min_b = min_ab
            # Normalização das entradas e saídas
            x_train_b = pre_processing_b.normalize_data(input_train_b,min_b,max_b)
            y_train_b = x_train_b
            x_test_b = pre_processing_b.normalize_data(input_test_b,min_b,max_b)
            y_test_b = x_test_b
        
            # Autoencoder para Rede B
            auto_encoder_b = Autoencoder(x_train_b, x_test_b, y_train_b, y_test_b, data_length_b, data_train_num_b, data_test_num_b, data_dimension_b)
            decoded_train_b, decoded_test_b, loss_train_b, loss_test_b = auto_encoder_b.convolutional_autoencoder_1d(kernel_size_b, epochs_num_b, optimizer_b, loss_b, cae_name_b)
            decoded_train_b_original = decoded_train_b*(max_b-min_b)+min_b
            decoded_test_b_original = decoded_test_b*(max_b-min_b)+min_b
            train_error_b = input_train_b - decoded_train_b_original
            test_error_b = input_test_b - decoded_test_b_original
        
            # Salva os resultados em um arquivo .csv
            np.savetxt('C:\\repos\\cae\\results\\input_train_b.csv', input_train_b, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\input_test_b.csv', input_test_b, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\decoded_train_b.csv', decoded_train_b_original, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\decoded_test_b.csv', decoded_test_b_original, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\train_error_b.csv', train_error_b, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\test_error_b.csv', test_error_b, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\loss_train_b.csv', loss_train_b, delimiter=',', fmt='%s')
            np.savetxt('C:\\repos\\cae\\results\\loss_test_b.csv', loss_test_b, delimiter=',', fmt='%s')
        
        else:
            pass