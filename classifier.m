clc;
clear;
close all;
% Carrega a base de dados e os resultados
data_original = csvread('C:\repos\cae\results\all_data_sample.csv');
data_decoded_a_0_13 = csvread('C:\repos\cae\results\all_data_decoded_a_0_13.csv');
data_decoded_a_14_17 = csvread('C:\repos\cae\results\all_data_decoded_a_14_17.csv');
data_decoded_a_18_37 = csvread('C:\repos\cae\results\all_data_decoded_a_18_37.csv');
data_decoded_a_38_45 = csvread('C:\repos\cae\results\all_data_decoded_a_38_45.csv');
data_decoded_b_0_13 = csvread('C:\repos\cae\results\all_data_decoded_b_0_13.csv');
data_decoded_b_14_17 = csvread('C:\repos\cae\results\all_data_decoded_b_14_17.csv');
data_decoded_b_18_37 = csvread('C:\repos\cae\results\all_data_decoded_b_18_37.csv');
data_decoded_b_38_45 = csvread('C:\repos\cae\results\all_data_decoded_b_38_45.csv');
data_decoded_a = [data_decoded_a_0_13;data_decoded_a_14_17;data_decoded_a_18_37;data_decoded_a_38_45];
data_decoded_b = [data_decoded_b_0_13;data_decoded_b_14_17;data_decoded_b_18_37;data_decoded_b_38_45];
real_labels_a = zeros(18,1);
real_labels_b = ones(28,1);
real_labels = [real_labels_a; real_labels_b];
predicted_labels = zeros(46,1);

% Identificação dos pacientes
sample_name_a = [100, 101, 103, 105, 106, 111, 117, 118, 121, 122, 123, 124, 205, 215, 220, 223, 230, 234];
sample_name_b = [107, 108, 109, 112, 113, 114, 115, 116, 119, 200, 201, 202, 203, 207, 208, 209, 210, 212, 213, 214, 217, 219, 221, 222, 228, 231, 232, 233];
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
