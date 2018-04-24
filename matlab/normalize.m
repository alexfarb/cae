clc;
close all;
clear all;

folder = 'C:\repos\ann-arrhythmia\diag\all\';
files = dir([folder '*.csv']);

for i=1:length(files)
   load([folder files(i).name]);
end

all_vectors = [X100m;X101m;X103m;X105m;X106m;X107m;X108m;X109m;X111m;
                X112m;X113m;X114m;X115m;X116m;X117m;X118m;X119m;X121m;
                X122m;X123m;X124m;X200m;X201m;X202m;X203m;X205m;X207m;
                X208m;X209m;X210m;X212m;X213m;X214m;X215m;X217m;X219m;
                X220m;X221m;X222m;X223m;X228m;X230m;X231m;X232m;X233m;
                X234m;];

norm_all = all_vectors - min(all_vectors(:));
norm_all = norm_all ./ max(norm_all(:));

x = 46;
outputname = 'C:\repos\ann-arrhythmia\diag\normalized\all\234m.csv';
csvwrite(outputname,norm_all(x:x,1:3600));