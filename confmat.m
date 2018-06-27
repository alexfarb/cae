clc;
clear;

% normal_known = [0, 0, 0, 0, 0, 0, 0, 0, 0];
% normal_predict = [0, 1, 0, 0, 0, 0, 0, 0, 0];
% anormal_known = [1, 1, 1, 1, 1, 1, 1, 1, 1];
% anormal_predict = [1, 1, 1, 1, 1, 0, 1, 1, 1];

normal_known = [0, 0, 0, 0, 0, 0, 0, 0, 0];
normal_predict = [0, 1, 0, 0, 0, 0, 0, 0, 0];
anormal_known = [1, 1, 1, 1, 1, 1, 1, 1, 1];
anormal_predict = [1, 1, 1, 1, 1, 0, 1, 1, 1];

all_known = [normal_known anormal_known];
all_predict = [normal_predict anormal_predict];

figure(1)
plotconfusion(all_known,all_predict);
set(gca,'xticklabel',{'Normal' 'Arritmia' ''})
set(gca,'yticklabel',{'Normal' 'Arritmia' ''})