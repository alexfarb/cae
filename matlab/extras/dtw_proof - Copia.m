clc;
clear;
close all;

data_original = csvread('100m_data_reshaped.csv');
data_decoded_a = csvread('100m_data_decoded_a.csv');
data_decoded_b = csvread('100m_data_decoded_b.csv');

dtw_original_a = dtw(data_original, data_decoded_a);
dtw_original_b = dtw(data_original, data_decoded_b);

figure();
subplot(3,1,1);
plot(data_original);
xlim([0 3600]);
title('Sinal original');
subplot(3,1,2);
plot(data_decoded_a);
xlim([0 3600]);
title(['Sinal Reconstruído pela Rede A. DTW = ' num2str(dtw_original_a) '']);
subplot(3,1,3);
plot(data_decoded_b);
xlim([0 3600]);
title(['Sinal Reconstruído pela Rede B. DTW = ' num2str(dtw_original_b) '']);