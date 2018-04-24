clc;
close all;
clear;

% Leitura da base de dados
filename_std = 'C:\repos\cae\data\new\healthy_samples_std.csv';
x_std = csvread(filename_std); % Entradas sem normalização
y = x_std; % Saida igual a entrada sem normalização

% % Escalonamento
% valmin = 0;
% valmax = 1;
% xmin = min(x_std);
% xmax = max(x_std);
% [lin,col]=size(x_std);
% x_norm = repmat((valmax-valmin)*ones(1,col)./(xmax-xmin),lin,1).*(x_std-repmat(xmin,lin,1))+ valmin*ones(lin,col);    

% Rede Neural Convolucional
layers = [
    imageInputLayer([1 3600])  
    convolution2dLayer([1 3],16,'Padding',1) 
    maxPooling2dLayer([1 2],'Stride',2) 
    convolution2dLayer([1 3],32,'Padding',1)
    imageInputLayer([1 3600])];
options = trainingOptions('sgdm');
net = trainNetwork(x_std,layers,options);
% Divisão das Amostras entre Treino e Teste
% x_train = x_norm(1:1200,1:27);
% x_val = x_norm(1:1200,28:54);
% y_train = y(1:1200,1:27);
% y_val = y(1:1200,28:54);
% 
% % Rede Convolucional
% inputLayer = imageInputLayer([n 1]);
% c1=convolution2dLayer([1 200],20,'stride',1);
% p1=maxPooling2dLayer([1 20],'stride',10);
% c2=convolution2dLayer([20 30],400,'numChannels',20);
% p2=maxPooling2dLayer([1 10],'stride',[1 2]);
% outputLayer = imageInputLayer([n 1]);
% convnet=[inputLayer; c1; p1; c2; p2; outputLayer]
% opts = trainingOptions('sgdm');
% convnet = trainNetwork(x_train,y_train,convnet,opts);
% net=newff(x_norm, y, hidden_layer_size); % Criação da RNA Feed-Forward utilizando para Treinamento o algoritmo  de Levenberg
% net.dividefcn='divideind'; % Funçao divisao de indices
% net.trainFcn = 'trainscg';
% net.divideparam.trainind=1:27; % Conjunto de Treinamento
% net.divideparam.valind=28:54; % Conjunto de Validação
% net.trainParam.epochs=1000;
% net.trainParam.lr = 0.01;
% [net,tr] = train(net,x_norm,y,'useGPU','yes'); % Treina a Rede Neural
% output_train = sim(net,x_train,'useGPU','yes'); % Simulação da Rede Neural. Contém a Saída obtida pela Rede Neural no Conjunto de Treinamento.
% output_val = sim(net,x_val,'useGPU','yes');
% error_train = abs(output_train - y_train);
% error_val = abs(output_val - y_val);

% % Resultados 100, 101 e 103
% x100m = x_std(1:1200,1:3);
% x100m = x100m(:);
% x100m = x100m';
% x100mrec = output_train(1:1200,1:3);
% x100mrec = x100mrec(:);
% x100mrec = x100mrec';
% x100merror = error_train(1:1200,1:3);
% x100merror = x100merror(:);
% x100merror = x100merror';
% 
% x101m = x_std(1:1200,4:6);
% x101m = x101m(:);
% x101m = x101m';
% x101mrec = output_train(1:1200,4:6);
% x101mrec = x101mrec(:);
% x101mrec = x101mrec';
% x101merror = error_train(1:1200,4:6);
% x101merror = x101merror(:);
% x101merror = x101merror';
% 
% x103m = x_std(1:1200,7:9);
% x103m = x103m(:);
% x103m = x103m';
% x103mrec = output_train(1:1200,7:9);
% x103mrec = x103mrec(:);
% x103mrec = x103mrec';
% x103merror = error_train(1:1200,7:9);
% x103merror = x103merror(:);
% x103merror = x103merror';
% 
% figure();
% subplot(3,3,1);
% plot(x100m);
% xlim([0 3600]);
% title(['Sinal Original 100m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,2);
% plot(x100mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 100m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,3);
% plot(x100merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 100m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,4);
% plot(x101m);
% xlim([0 3600]);
% title(['Sinal Original 101m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,5);
% plot(x101mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 101m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,6);
% plot(x101merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 101m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,7);
% plot(x103m);
% xlim([0 3600]);
% title(['Sinal Original 103m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,8);
% plot(x103mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 103m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,9);
% plot(x103merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 103m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% 
% % Resultados 105, 106 e 111
% x105m = x_std(1:1200,10:12);
% x105m = x105m(:);
% x105m = x105m';
% x105mrec = output_train(1:1200,10:12);
% x105mrec = x105mrec(:);
% x105mrec = x105mrec';
% x105merror = error_train(1:1200,10:12);
% x105merror = x105merror(:);
% x105merror = x105merror';
% 
% x106m = x_std(1:1200,13:15);
% x106m = x106m(:);
% x106m = x106m';
% x106mrec = output_train(1:1200,13:15);
% x106mrec = x106mrec(:);
% x106mrec = x106mrec';
% x106merror = error_train(1:1200,13:15);
% x106merror = x106merror(:);
% x106merror = x106merror';
% 
% x111m = x_std(1:1200,16:18);
% x111m = x111m(:);
% x111m = x111m';
% x111mrec = output_train(1:1200,16:18);
% x111mrec = x111mrec(:);
% x111mrec = x111mrec';
% x111merror = error_train(1:1200,16:18);
% x111merror = x111merror(:);
% x111merror = x111merror';
% 
% figure();
% subplot(3,3,1);
% plot(x105m);
% xlim([0 3600]);
% title(['Sinal Original 105m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,2);
% plot(x105mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 105m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,3);
% plot(x105merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 105m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,4);
% plot(x106m);
% xlim([0 3600]);
% title(['Sinal Original 106m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,5);
% plot(x106mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 106m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,6);
% plot(x106merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 106m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,7);
% plot(x111m);
% xlim([0 3600]);
% title(['Sinal Original 111m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,8);
% plot(x111mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 111m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,9);
% plot(x111merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 111m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% 
% % Resultados 117, 118 e 121
% x117m = x_std(1:1200,19:21);
% x117m = x117m(:);
% x117m = x117m';
% x117mrec = output_train(1:1200,19:21);
% x117mrec = x117mrec(:);
% x117mrec = x117mrec';
% x117merror = error_train(1:1200,19:21);
% x117merror = x117merror(:);
% x117merror = x117merror';
% 
% x118m = x_std(1:1200,22:24);
% x118m = x118m(:);
% x118m = x118m';
% x118mrec = output_train(1:1200,22:24);
% x118mrec = x118mrec(:);
% x118mrec = x118mrec';
% x118merror = error_train(1:1200,22:24);
% x118merror = x118merror(:);
% x118merror = x118merror';
% 
% x121m = x_std(1:1200,25:27);
% x121m = x121m(:);
% x121m = x121m';
% x121mrec = output_train(1:1200,25:27);
% x121mrec = x121mrec(:);
% x121mrec = x121mrec';
% x121merror = error_train(1:1200,25:27);
% x121merror = x121merror(:);
% x121merror = x121merror';
% 
% figure();
% subplot(3,3,1);
% plot(x117m);
% xlim([0 3600]);
% title(['Sinal Original 117m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,2);
% plot(x117mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 117m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,3);
% plot(x117merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 117m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,4);
% plot(x118m);
% xlim([0 3600]);
% title(['Sinal Original 118m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,5);
% plot(x118mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 118m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,6);
% plot(x118merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 118m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,7);
% plot(x121m);
% xlim([0 3600]);
% title(['Sinal Original 121m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,8);
% plot(x121mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 121m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,9);
% plot(x111merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 121m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% 
% % Resultados 122, 123 e 124
% x122m = x_std(1:1200,28:30);
% x122m = x122m(:);
% x122m = x122m';
% x122mrec = output_val(1:1200,1:3);
% x122mrec = x122mrec(:);
% x122mrec = x122mrec';
% x122merror = error_val(1:1200,1:3);
% x122merror = x122merror(:);
% x122merror = x122merror';
% 
% x123m = x_std(1:1200,31:33);
% x123m = x123m(:);
% x123m = x123m';
% x123mrec = output_val(1:1200,4:6);
% x123mrec = x123mrec(:);
% x123mrec = x123mrec';
% x123merror = error_val(1:1200,4:6);
% x123merror = x123merror(:);
% x123merror = x123merror';
% 
% x124m = x_std(1:1200,34:36);
% x124m = x124m(:);
% x124m = x124m';
% x124mrec = output_val(1:1200,7:9);
% x124mrec = x124mrec(:);
% x124mrec = x124mrec';
% x124merror = error_val(1:1200,7:9);
% x124merror = x124merror(:);
% x124merror = x124merror';
% 
% figure();
% subplot(3,3,1);
% plot(x122m);
% xlim([0 3600]);
% title(['Sinal Original 122m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,2);
% plot(x122mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 122m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,3);
% plot(x122merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 122m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,4);
% plot(x123m);
% xlim([0 3600]);
% title(['Sinal Original 123m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,5);
% plot(x123mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 123m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,6);
% plot(x123merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 123m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,7);
% plot(x124m);
% xlim([0 3600]);
% title(['Sinal Original 124m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,8);
% plot(x124mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 124m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,9);
% plot(x124merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 124m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% 
% % Resultados 205, 215 e 220
% x205m = x_std(1:1200,37:39);
% x205m = x205m(:);
% x205m = x205m';
% x205mrec = output_val(1:1200,10:12);
% x205mrec = x205mrec(:);
% x205mrec = x205mrec';
% x205merror = error_val(1:1200,10:12);
% x205merror = x205merror(:);
% x205merror = x205merror';
% 
% x215m = x_std(1:1200,40:42);
% x215m = x215m(:);
% x215m = x215m';
% x215mrec = output_val(1:1200,13:15);
% x215mrec = x215mrec(:);
% x215mrec = x215mrec';
% x215merror = error_val(1:1200,13:15);
% x215merror = x215merror(:);
% x215merror = x215merror';
% 
% x220m = x_std(1:1200,43:45);
% x220m = x220m(:);
% x220m = x220m';
% x220mrec = output_val(1:1200,16:18);
% x220mrec = x220mrec(:);
% x220mrec = x220mrec';
% x220merror = error_val(1:1200,16:18);
% x220merror = x220merror(:);
% x220merror = x220merror';
% 
% figure();
% subplot(3,3,1);
% plot(x205m);
% xlim([0 3600]);
% title(['Sinal Original 205m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,2);
% plot(x205mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 205m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,3);
% plot(x205merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 205m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,4);
% plot(x215m);
% xlim([0 3600]);
% title(['Sinal Original 215m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,5);
% plot(x215mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 215m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,6);
% plot(x215merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 215m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,7);
% plot(x220m);
% xlim([0 3600]);
% title(['Sinal Original 220m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,8);
% plot(x220mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 220m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,9);
% plot(x220merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 220m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% 
% % Resultados 223, 230 e 234
% x223m = x_std(1:1200,46:48);
% x223m = x223m(:);
% x223m = x223m';
% x223mrec = output_val(1:1200,19:21);
% x223mrec = x223mrec(:);
% x223mrec = x223mrec';
% x223merror = error_val(1:1200,19:21);
% x223merror = x223merror(:);
% x223merror = x223merror';
% 
% x230m = x_std(1:1200,49:51);
% x230m = x230m(:);
% x230m = x230m';
% x230mrec = output_val(1:1200,22:24);
% x230mrec = x230mrec(:);
% x230mrec = x230mrec';
% x230merror = error_val(1:1200,22:24);
% x230merror = x230merror(:);
% x230merror = x230merror';
% 
% x234m = x_std(1:1200,52:54);
% x234m = x234m(:);
% x234m = x234m';
% x234mrec = output_val(1:1200,25:27);
% x234mrec = x234mrec(:);
% x234mrec = x234mrec';
% x234merror = error_val(1:1200,25:27);
% x234merror = x234merror(:);
% x234merror = x234merror';
% 
% figure();
% subplot(3,3,1);
% plot(x223m);
% xlim([0 3600]);
% title(['Sinal Original 223m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,2);
% plot(x223mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 223m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,3);
% plot(x223merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 223m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,4);
% plot(x230m);
% xlim([0 3600]);
% title(['Sinal Original 230m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,5);
% plot(x230mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 230m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,6);
% plot(x230merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 230m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,7);
% plot(x234m);
% xlim([0 3600]);
% title(['Sinal Original 234m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,8);
% plot(x234mrec);
% xlim([0 3600]);
% title(['Sinal Reconstruído 234m. Camada Oculta = ' num2str(hidden_layer_size) '.']);
% subplot(3,3,9);
% plot(x234merror);
% xlim([0 3600]);
% title(['Erro entre o Sinal 234m Original e o Reconstruído. Camada Oculta = ' num2str(hidden_layer_size) '.']);