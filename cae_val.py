import warnings
warnings.filterwarnings("ignore")
import numpy as np
from keras.models import load_model
from keras import backend as K
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class PreProcessingData(object):
    # Metódo Construtor
    def __init__(self, min, max):
        self.min = min
        self.max = max

    # função para a normalização dos dados
    def normalize_data(self, data):
        return (data - self.min) / (self.max - self.min)  # Cálculo para a normalização dos dados

class Autoencoder(object):
    def __init__(self):
        pass
    # Rede A
    def autoencoder(self, data, min, max, autoencoder_option):
        if autoencoder_option == 0:
            model = load_model('C:\\repos\\cae\\saved_models\\cae_a.h5')
        elif autoencoder_option ==1:
            model = load_model('C:\\repos\\cae\\saved_models\\cae_b.h5')
        data_proc = PreProcessingData(min, max)
        data_original = data
        data_norm = data_proc.normalize_data(data_original)
        data_reshaped = np.expand_dims(data_norm, axis=2)
        z = data_reshaped.T
        data_input = np.expand_dims(z, axis=2)
        predict_model_a = model.predict(data_input)
        decoded_reshaped = (predict_model_a.reshape(1, 3600))
        data_decoded = decoded_reshaped*(max - min) + min
        return data_decoded
    
    def plot_reconstruct(self, sample_name, data_reshaped, data_decode_a, data_decode_b, msq_a, msq_b, predict_label, correct_label):
        title = "Paciente %s, Classificado = %s, Categoria Real = %s %%" %(sample_name, predict_label, correct_label)
        plt.suptitle(title)
        plt.subplot(311)
        plt.plot(data_reshaped)
        plt.title("Sinal Original")
        plt.xlim(0, 3600)
        plt.subplot(312)
        plt.plot(data_decode_a)
        plt.title("Sinal Reconstruído pela Rede A, EQM = %s" (msq_a))
        plt.xlim(0, 3600)
        plt.subplot(313)
        plt.plot(data_decode_b)
        plt.title("Sinal Reconstruído pela Rede B, EQM = %s" (msq_b))
        plt.xlim(0, 3600)
        plt.show()
    

dataset_a = np.loadtxt("C:\\repos\\cae\\data\\conv1d\\healthy_samples_std.csv",delimiter=",") # Carrega a Base de Dados A
dataset_b = np.loadtxt("C:\\repos\\cae\\data\\conv1d\\abnormal_samples_std.csv",delimiter=",") # Carrega a Base de Dados B
dataset_a_labels = np.zeros((len(dataset_a[:,0]),1)) # Categorias da Base de Dados A (0)
dataset_b_labels = np.ones((len(dataset_b[:,0]),1)) # Categorias da base de Dados B (1)
# Adiciona as Categorias as Matrizes
data_a = np.concatenate((dataset_a,dataset_a_labels),axis=1)
data_b = np.concatenate((dataset_b,dataset_b_labels),axis=1)
# Junta todos os dados em uma matriz
data = np.concatenate((data_a,data_b),axis=0)
data_sample = np.delete(data, -1, axis=1)
data_label = data[:,-1]
data_label = np.expand_dims(data_label, axis=2)
max = 1439
min = 572
# Predição usando o autoencoder
predict_label = [] # Vetor para alocar as categorias preditas
correct_label = [] # Vetor para alocar as categorias preditas corretamente (0 = Certo, 1 = Errado)

sample_name_a = ['X100m', 'X101m', 'X103m', 'X105m', 'X106m', 'X111m', 'X117m', 
                 'X118m', 'X121m', 'X122m', 'X123m', 'X124m', 'X205m', 'X215m', 
                 'X220m', 'X223m', 'X230m', 'X234m']
sample_name_b= ['X107m','X108m','X109m','X112m','X113m','X114m',
                    'X115m','X116m','X119m','X200m','X201m','X202m',
                    'X203m','X207m','X208m','X209m','X210m','X212m',
                    'X213m','X214m','X217m','X219m','X221m','X222m',
                    'X228m', 'X231m', 'X232m', 'X233m']

sample_name = np.concatenate((sample_name_a,sample_name_b),axis=0)

data_array = [14,15,16,17,38,39,40,41,42,43,44,45]

cae = Autoencoder()
i = 0  
for data_index in data_array:
    data_decode_a = cae.autoencoder(data_sample[data_index,:], min, max, 0)
    data_dim = np.expand_dims(data_sample[data_index,:], axis=2)
    data_reshaped = data_dim.T
    msq_a = mean_squared_error(data_reshaped, data_decode_a)
    data_decode_b = cae.autoencoder(data_sample[data_index,:], min, max, 1)
    msq_b = mean_squared_error(data_reshaped, data_decode_b)
    
    
    if msq_a < msq_b:
        predict_label.append(0)
    elif msq_a > msq_b:
        predict_label.append(1)
        
    if data_label[data_index,:] == predict_label[i]:
        correct_label.append(0)
    elif data_label[data_index,:] != predict_label[i]:
        correct_label.append(1)
        
    cae.plot_reconstruct(sample_name[data_index], data_reshaped, data_decode_a, data_decode_b, msq_a, msq_b, predict_label[i], correct_label[i])
    i = i+1
    
#print(error_model_a)
#plt.plot(a)
#plt.plot(a_recon)
#plt.show()

#predict_model_b = model_b.predict(dataset_a_reshaped[0])
#error_model_b = losses.mean_squared_error(dataset_a[0], predict_model_b)
#print(error_model_b)