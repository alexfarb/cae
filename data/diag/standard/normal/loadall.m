clear;

files = dir('*.csv');
for i=1:length(files)
    eval(['load ' files(i).name ' -ascii']);
end
all_vectors = [X100m;X101m;X103m;X105m;X106m;X112m;
                    X113m;X114m;X115m;X116m;X117m;X121m;
                    X122m;X123m;X201m;X202m;X205m;X209m;
                    X213m;X215m;X219m;X220m;X234m];
abnormal_samples_std = all_vectors;

csvwrite('C:\repos\cae\data\conv1d\normal_samples_std.csv',abnormal_samples_std);