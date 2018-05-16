import numpy as np
import matplotlib.pyplot as plt

# Classe para os Plots
class Plots(object):
    def __init__(self, loss_train, loss_test):
        self.loss_train = loss_train
        self.loss_test = loss_test

    def plot_reconstruct_train(self, sample_name, input_train, decoded_train, train_error, data_row, data_length):
        title = "Paciente %s (Conjunto de Treino)" % (sample_name)
        plt.suptitle(title)
        plt.subplot(311)
        plt.plot(input_train[data_row, :])
        plt.title("Sinal Original")
        plt.xlim(0, data_length)
        plt.subplot(312)
        plt.plot(decoded_train[data_row, :])
        plt.title("Sinal Reconstruído")
        plt.xlim(0, data_length)
        plt.subplot(313)
        plt.plot(train_error[data_row, :])
        plt.title("Erro entre o Sinal Original e o Reconstruído")
        plt.xlim(0, data_length)
        plt.show()

    def plot_reconstruct_test(self, sample_name, input_test, decoded_test, test_error, data_row, data_length):
        title = "Paciente %s (Conjunto de Teste)" % (sample_name)
        plt.suptitle(title)
        plt.subplot(311)
        plt.plot(input_test[data_row, :])
        plt.title("Sinal Original")
        plt.xlim(0, data_length)
        plt.subplot(312)
        plt.plot(decoded_test[data_row, :])
        plt.title("Sinal Reconstruído")
        plt.xlim(0, data_length)
        plt.subplot(313)
        plt.plot(test_error[data_row, :])
        plt.title("Erro entre o Sinal Original e o Reconstruído")
        plt.xlim(0, data_length)
        plt.show()

    def loss_curve(self):
        epochs = len(self.loss_train.T)
        title = "Erro Médio Quadrático ao longo das Épocas"
        plt.suptitle(title)
        plt.subplot(211)
        plt.plot(self.loss_train.T)
        plt.title("Erro Médio Quadrático (Treino)")
        plt.xlim(0, epochs)
        plt.subplot(212)
        plt.plot(self.loss_test.T)
        plt.title("Erro Médio Quadrático (Teste)")
        plt.xlim(0, epochs)
        plt.show()


option = 2 # 1 para a Rede A e 2 para a Rede B

if option == 1:
    # Carregamento dos Resultados da Rede A
    input_train_a = np.loadtxt('C:\\repos\\cae\\results\\input_train_a.csv', delimiter=",")
    input_test_a = np.loadtxt('C:\\repos\\cae\\results\\input_test_a.csv', delimiter=",")
    decoded_train_a = np.loadtxt('C:\\repos\\cae\\results\\decoded_train_a.csv', delimiter=",")
    decoded_test_a = np.loadtxt('C:\\repos\\cae\\results\\decoded_test_a.csv', delimiter=",")
    train_error_a = np.loadtxt('C:\\repos\\cae\\results\\train_error_a.csv', delimiter=",")
    test_error_a = np.loadtxt('C:\\repos\\cae\\results\\test_error_a.csv', delimiter=",")
    loss_train_a = np.loadtxt('C:\\repos\\cae\\results\\loss_train_a.csv', delimiter=",")
    loss_test_a = np.loadtxt('C:\\repos\\cae\\results\\loss_test_a.csv', delimiter=",")
    data_length = 3600

    # Plots para a Rede A
    plot_a = Plots(loss_train_a, loss_test_a)

    sample_name_a = ['X100m', 'X101m', 'X103m', 'X105m', 'X106m', 'X111m', 'X117m', 'X118m', 'X121m', 'X122m', 'X123m', 'X124m', 'X205m', 'X215m', 'X220m', 'X223m', 'X230m', 'X234m']

    for i in range(0,len(input_train_a[:,0])):
        plot_a.plot_reconstruct_train(sample_name_a[i], input_train_a, decoded_train_a, train_error_a, i, data_length)

    for i in range(0,len(input_test_a[:,0])):
        plot_a.plot_reconstruct_test(sample_name_a[len(input_train_a[:,0])+i], input_train_a, decoded_train_a, train_error_a, i, data_length)

    plot_a.loss_curve()

elif option == 2:
    # Carregamento dos Resultados da Rede B
    input_train_b = np.loadtxt('C:\\repos\\cae\\results\\input_train_b.csv', delimiter=",")
    input_test_b = np.loadtxt('C:\\repos\\cae\\results\\input_test_b.csv', delimiter=",")
    decoded_train_b = np.loadtxt('C:\\repos\\cae\\results\\decoded_train_b.csv', delimiter=",")
    decoded_test_b = np.loadtxt('C:\\repos\\cae\\results\\decoded_test_b.csv', delimiter=",")
    train_error_b = np.loadtxt('C:\\repos\\cae\\results\\train_error_b.csv', delimiter=",")
    test_error_b = np.loadtxt('C:\\repos\\cae\\results\\test_error_b.csv', delimiter=",")
    loss_train_b = np.loadtxt('C:\\repos\\cae\\results\\loss_train_b.csv', delimiter=",")
    loss_test_b = np.loadtxt('C:\\repos\\cae\\results\\loss_test_b.csv', delimiter=",")
    data_length = 3600

    # Plots para a Rede A
    plot_b = Plots(loss_train_b, loss_test_b)

    sample_name_b= ['X107m','X108m','X109m','X112m','X113m','X114m',
                    'X115m','X116m','X119m','X200m','X201m','X202m',
                    'X203m','X207m','X208m','X209m','X210m','X212m',
                    'X213m','X214m','X217m','X219m','X221m','X222m',
                    'X228m', 'X231m', 'X232m', 'X233m']

    for i in range(0, len(input_train_b[:, 0])):
        plot_b.plot_reconstruct_train(sample_name_b[i], input_train_b, decoded_train_b, train_error_b, i, data_length)

    for i in range(0, len(input_test_b[:, 0])):
        plot_b.plot_reconstruct_test(sample_name_b[len(input_train_b[:, 0]) + i], input_train_b, decoded_train_b,
                                     train_error_b, i, data_length)

    plot_b.loss_curve()

else:
    pass
