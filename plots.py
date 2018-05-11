import numpy as np
import matplotlib.pyplot as plt

# Classe para os Plots
class Plots(object):
    def __init__(self, input_train, input_test, decoded_train, decoded_test, train_error, test_error, loss_train, loss_test):
        self.input_data = input_data
        self.decoded_data = decoded_data

    def plot_reconstruct(self, sample_name, data_row, data_length):
        title = "Paciente %s" % (sample_name)
        plt.suptitle(title)
        plt.subplot(311)
        plt.plot(self.input_data[data_row, :])
        plt.title("Sinal Original")
        plt.xlim(0, data_length)
        plt.subplot(312)
        plt.plot(self.decoded_data[data_row, :])
        plt.title("Sinal Reconstruído")
        plt.xlim(0, data_length)
        plt.subplot(313)
        plt.plot(train_error[data_row, :])
        plt.title("Erro entre o Sinal Original e o Reconstruído")
        plt.xlim(0, data_length)
        plt.show()

# Carregamento dos Resultados da Rede A
input_train_a = np.loadtxt('C:\\repos\\cae\\results\\input_train_a.csv', delimiter=",")
input_test_a = np.loadtxt('C:\\repos\\cae\\results\\input_test_a.csv', delimiter=",")
decoded_train_a_original = np.loadtxt('C:\\repos\\cae\\results\\decoded_train_a.csv', delimiter=",")
decoded_test_a_original = np.loadtxt('C:\\repos\\cae\\results\\decoded_test_a.csv', delimiter=",")
train_error_a = np.loadtxt('C:\\repos\\cae\\results\\train_error_a.csv', delimiter=",")
test_error_a = np.loadtxt('C:\\repos\\cae\\results\\test_error_a.csv', delimiter=",")
loss_train_a = np.loadtxt('C:\\repos\\cae\\results\\loss_train_a.csv', delimiter=",")
loss_test_a = np.loadtxt('C:\\repos\\cae\\results\\loss_test_a.csv', delimiter=",")
data_length = 3600

# Plots para a Rede A
# plot = Plots(input_train_a, decoded_train_a_original)
#     plot.plot_reconstruct(100, train_error_a, 0, data_length_a)
#     # Plots para a Rede B
#     plot = Plots(input_train_b, decoded_train_b_original)
#     plot.plot_reconstruct(107, train_error_b, 0, data_length_b)