clc;
clear;
close all;

data_original = csvread('C:\repos\cae\results\all_data_sample.csv');
data_decoded_a_0_13 = csvread('C:\repos\cae\results\all_data_decoded_a_0_13.csv');
data_decoded_a_14_17 = csvread('C:\repos\cae\results\all_data_decoded_a_14_17.csv');
data_decoded_a_18_37 = csvread('C:\repos\cae\results\all_data_decoded_a_18_37.csv');
data_decoded_a_38_45 = csvread('C:\repos\cae\results\all_data_decoded_a_38_45.csv');
data_decoded_b_0_13 = csvread('C:\repos\cae\results\all_data_decoded_b_0_13.csv');
data_decoded_b_14_17 = csvread('C:\repos\cae\results\all_data_decoded_b_14_17.csv');
data_decoded_b_18_37 = csvread('C:\repos\cae\results\all_data_decoded_b_18_37.csv');
data_decoded_b_38_45 = csvread('C:\repos\cae\results\all_data_decoded_b_38_45.csv');

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