clc;
close all;
clear all;

% Leitura da base de dados
filename_std = 'C:\repos\ann-arrhythmia\dataproc\healthy_samples.csv';
filename_norm = 'C:\repos\ann-arrhythmia\dataproc\healthy_samples_norm.csv';

x_norm = csvread(filename_norm); % Entradas Normalizadas
x_std = csvread(filename_std); % Entradas sem normalização
y = x_std; % Saida igual a entrada desnormalizada
n = length(x_std);

% Divisão das Amostras entre Treino e Teste
x_train = x_norm(1:1200,1:27);
x_val = x_norm(1:1200,28:54);
y_train = y(1:1200,1:27);
y_val = y(1:1200,28:54);

hiddenSize = 10;
autoenc1 = trainAutoencoder(x_train,hiddenSize,...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.05,...
    'DecoderTransferFunction','purelin');

features1 = encode(autoenc1,x_train);