import numpy as np
from keras import losses
from keras.models import load_model

dataset_a = np.loadtxt("C:\\repos\\cae\\data\\conv1d\\healthy_samples_std.csv",delimiter=",") # Carrega a Base de Dados A
dataset_b = np.loadtxt("C:\\repos\\cae\\data\\conv1d\\abnormal_samples_std.csv",delimiter=",") # Carrega a Base de Dados B
dataset_a_labels = np.zeros((len(dataset_a[:,0]),1)) # Categorias da Base de Dados A (0)
dataset_b_labels = np.ones((len(dataset_b[:,0]),1)) # Categorias da base de Dados B (1)
# Adiciona as Categorias as Matrizes
data_a = np.concatenate((dataset_a,dataset_a_labels),axis=1)
data_b = np.concatenate((dataset_b,dataset_b_labels),axis=1)
# Junta todos os dados em uma matriz
data = np.concatenate((data_a,data_b),axis=0)
# Carregamento dos modelos
model_a = load_model('C:\\repos\\cae\\saved_models\\cae_a.h5')
model_b = load_model('C:\\repos\\cae\\saved_models\\cae_b.h5')
# Classificação das Amostras
a = dataset_a[0,:]
x = np.expand_dims(a, axis=2)
z = x.T
y = np.expand_dims(z, axis=2)
predict_model_a = model_a.predict(y)
a_reshaped = (predict_model_a.reshape(1, 3600))
error_model_a = z-a_reshaped
print(error_model_a)
#predict_model_b = model_b.predict(dataset_a_reshaped[0])
#error_model_b = losses.mean_squared_error(dataset_a[0], predict_model_b)
#print(error_model_b)