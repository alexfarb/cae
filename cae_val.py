import warnings
warnings.filterwarnings("ignore")
import numpy as np
from keras.models import load_model
from keras import backend as K
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# from dtw import dtw
# from numpy.linalg import norm
# from scipy.spatial.distance import euclidean

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
    
    def plot_reconstruct(self, sample_name, data_reshaped, data_decode_a, data_decode_b, msq_a, msq_b, predict_label, data_label):
        plt.cla()
        plt.clf()
        title = "Paciente %s, Classe (Predição) = %s, Classe (Real) = %s " %(sample_name, predict_label, int(data_label))
        plt.suptitle(title)
        plt.subplot(311)
        plt.plot(data_reshaped[0,:])
        plt.title("Sinal Original")
        plt.xlabel("Amostras")
        plt.ylabel("Amplitude(mV)")
        plt.xlim(0, len(data_reshaped.T))
        plt.subplot(312)
        plt.plot(data_decode_a[0,:])
        plt.title("Sinal Reconstruído pela Rede A, Erro Quadrático Médio = %s" %(np.round(msq_a,5)))
        plt.xlabel("Amostras")
        plt.ylabel("Amplitude(mV)")
        plt.xlim(0, len(data_reshaped.T))
        plt.subplot(313)
        plt.plot(data_decode_b[0,:])
        plt.title("Sinal Reconstruído pela Rede B, Erro Quadrático Médio = %s" %(np.round(msq_b,5)))
        plt.xlabel("Amostras")
        plt.ylabel("Amplitude(mV)")
        plt.xlim(0, len(data_reshaped.T))
        fig1 = plt.gcf()
        fig1_path = 'C:\\repos\\cae\\plots\\%s.png' % (sample_name)
        plt.draw()
        fig1.set_size_inches((16, 14), forward=False)
        fig1.savefig(fig1_path)
        plt.cla()
        plt.clf()
    
def main_cae_val():
    dataset_a = np.loadtxt("C:\\repos\\cae\\data\\conv1d\\normal_samples_std.csv",delimiter=",") # Carrega a Base de Dados A
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
    data_label_list = data_label.tolist()
    max = 1526
    min = 403
    # Predição usando o autoencoder
    predict_label = [] # Vetor para alocar as categorias preditas
    correct_label = [] # Vetor para alocar as categorias preditas corretamente (0 = Certo, 1 = Errado)

    sample_name_a = ['100m', '101m', '103m', '105m', '106m', '112m', '113m',
                     '114m', '115m', '116m', '117m', '121m', '122m', '123m',
                     '201m', '202m', '205m', '209m',
                     '213m', '215m', '219m', '220m', '234m']
    sample_name_b= ['107m','108m','109m','111m','118m','119m',
                        '124m','200m','203m','207m','208m','210m',
                        '212m','214m','217m','221m','222m','223m',
                        '228m','230m','231m','232m','233m']

    sample_name = np.concatenate((sample_name_a,sample_name_b),axis=0)

    # data_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # data_array = [14, 15, 16, 17, 18, 19, 20 , 21, 22]
    # data_array = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    data_array = [37, 38, 39, 40, 41, 42, 43, 44, 45]
    # data_array = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
    # data_array = [0]

    all_data_decoded_a = []
    all_data_decoded_b = []

    cae = Autoencoder()
    i = 0  
    for data_index in data_array:
        data_decoded_a = cae.autoencoder(data_sample[data_index,:], min, max, 0)
        data_dim = np.expand_dims(data_sample[data_index,:], axis=2)
        data_reshaped = data_dim.T
        msq_a = mean_squared_error(data_reshaped, data_decoded_a)
        data_decoded_b = cae.autoencoder(data_sample[data_index,:], min, max, 1)
        msq_b = mean_squared_error(data_reshaped, data_decoded_b)
        
        if msq_a < msq_b:
            predict_label.append(0)
        elif msq_a > msq_b:
            predict_label.append(1)

        if data_label[data_index,:] == predict_label[i]:
            correct_label.append(0)
        elif data_label[data_index,:] != predict_label[i]:
            correct_label.append(1)

        cae.plot_reconstruct(sample_name[data_index], data_reshaped, data_decoded_a,
                                data_decoded_b, msq_a, msq_b, predict_label[i],
                               data_label[data_index])


        data_decoded_a = np.reshape(data_decoded_a,3600)
        data_decoded_b = np.reshape(data_decoded_b, 3600)
        all_data_decoded_a.append(data_decoded_a)
        all_data_decoded_b.append(data_decoded_b)

        # np.savetxt('C:\\repos\\cae\\results\\100m_data_reshaped.csv', data_reshaped, delimiter=',', fmt='%s')
        # np.savetxt('C:\\repos\\cae\\results\\100m_data_decoded_a.csv', data_decode_a, delimiter=',', fmt='%s')
        # np.savetxt('C:\\repos\\cae\\results\\100m_data_decoded_b.csv', data_decode_b, delimiter=',', fmt='%s')
        i = i+1

    data_decoded_a = data_decoded_a
    data_decoded_b = data_decoded_b
    # print(data_decoded_a)
    # np.savetxt('C:\\repos\\cae\\results\\all_data_sample.csv', data_sample, delimiter=',', fmt='%s')
    np.savetxt('C:\\repos\\cae\\results\\all_data_decoded_a_37_45.csv', all_data_decoded_a, delimiter=',', fmt='%s')
    np.savetxt('C:\\repos\\cae\\results\\all_data_decoded_b_37_45.csv', all_data_decoded_b, delimiter=',', fmt='%s')
#     print(data_decoded_b)
    print(data_array)
    print(predict_label)
    print(correct_label)