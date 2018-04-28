clear;

files = dir('*.csv');
for i=1:length(files)
    eval(['load ' files(i).name ' -ascii']);
end
all_vectors = [X100m;X101m;X103m;X105m;X106m;X111m;
                    X117m;X118m;X121m;X122m;X123m;X124m;
                    X205m;X215m;X220m;X223m;X230m;X234m;];
    
healthy_samples_std = all_vectors;

csvwrite('C:\repos\cae\data\conv1d\healthy_samples_std.csv',healthy_samples_std);