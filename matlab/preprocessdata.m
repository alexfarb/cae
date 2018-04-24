clc;
clear all;
close all;


healthy_samples_norm = [];
for i = 1:18
    healthy_samples_norm =  [healthy_samples_norm organizedatahealthy(i)];
end

csvwrite('C:\repos\ann-arrhythmia\dataproc\healthy_samples_norm.csv',healthy_samples_norm);