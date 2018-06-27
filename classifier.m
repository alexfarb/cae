clc;
clear;
close all;
% Carrega a base de dados e os resultados
data_original = csvread('C:\repos\cae\results\all_data_sample.csv');
data_decoded_a_a = csvread('C:\repos\cae\results\all_data_decoded_a_0_13.csv');
data_decoded_a_b = csvread('C:\repos\cae\results\all_data_decoded_a_14_22.csv');
data_decoded_a_c = csvread('C:\repos\cae\results\all_data_decoded_a_23_36.csv');
data_decoded_a_d = csvread('C:\repos\cae\results\all_data_decoded_a_37_45.csv');
data_decoded_b_a = csvread('C:\repos\cae\results\all_data_decoded_b_0_13.csv');
data_decoded_b_b = csvread('C:\repos\cae\results\all_data_decoded_b_14_22.csv');
data_decoded_b_c = csvread('C:\repos\cae\results\all_data_decoded_b_23_36.csv');
data_decoded_b_d = csvread('C:\repos\cae\results\all_data_decoded_b_37_45.csv');
data_decoded_a = [data_decoded_a_a;data_decoded_a_b;data_decoded_a_c;data_decoded_a_d];
data_decoded_b = [data_decoded_b_a;data_decoded_b_b;data_decoded_b_c;data_decoded_b_d];
real_labels_a = zeros(23,1);
real_labels_b = ones(23,1);
real_labels = [real_labels_a; real_labels_b];
predicted_labels = zeros(46,1);

% Identificação dos pacientes
sample_name_a = [100, 101, 103, 105, 106, 112, 113, 114, 115, 116, 117, 121, 122, 123, 201, 202, 205, 209, 213, 215, 219, 220, 234];
sample_name_b = [107, 108, 109, 111, 118, 119, 124, 200, 203, 207, 208, 210, 212, 214, 217, 221, 222, 223, 228, 230, 231, 232, 233];
sample_name = [sample_name_a sample_name_b];

%% Plots

for i = 1:length(sample_name');                   
                    
    dtw_original_a = dtw(data_original(i,:), data_decoded_a(i,:));
    dtw_original_b = dtw(data_original(i,:), data_decoded_b(i,:));

    fig_a = figure('units','normalized','outerposition',[0 0 1 1]);
    suptitle(['Sobreposição Paciente ' num2str(sample_name(:,i)) 'm (Sinal Original X Reconstrução Rede A)']);
    dtw(data_original(i,:), data_decoded_a(i,:));
    fig_a_name = ['dtw_original_a_' num2str(sample_name(:,i)) ''];
    print(fig_a_name,'-dpng','-r300');

    fig_b = figure('units','normalized','outerposition',[0 0 1 1]);
    suptitle(['Sobreposição Paciente ' num2str(sample_name(:,i)) 'm (Sinal Original X Reconstrução Rede B)']);
    dtw(data_original(i,:), data_decoded_b(i,:));
    fig_b_name = ['dtw_original_b_' num2str(sample_name(:,i)) ''];
    print(fig_b_name,'-dpng','-r300');

    if dtw_original_a < dtw_original_b
        predicted_labels(i,:) = 0;    
    else
        predicted_labels(i,:) = 1;
    end

    fig_c = figure('units','normalized','outerposition',[0 0 1 1]);
    subplot(3,1,1);
    plot(data_original(i,:));
    xlabel('Amostras');
    ylabel('Amplitude(mV)');
    xlim([0 length(data_original(i,:))]);
    title('Sinal original');
    subplot(3,1,2);
    plot(data_decoded_a(i,:));
    xlabel('Amostras');
    ylabel('Amplitude(mV)');
    xlim([0 length(data_original(i,:))]);
    title(['Sinal Reconstruído pela Rede A. DTW = ' num2str(dtw_original_a) '']);
    subplot(3,1,3);
    plot(data_decoded_b(i,:));
    xlabel('Amostras');
    ylabel('Amplitude(mV)');
    xlim([0 length(data_original(i,:))]);
    title(['Sinal Reconstruído pela Rede B. DTW = ' num2str(dtw_original_b) '']);
    suptitle(['Paciente = ' num2str(sample_name(:,i)) 'm. Classe (Predição) = ' num2str(predicted_labels(i,:)) '. Classe (Real) = ' num2str(real_labels(i,:)) '']);
    fig_c_name = ['reconstruction_' num2str(sample_name(:,i)) ''];
    print(fig_c_name,'-dpng','-r300');

end

close all;
