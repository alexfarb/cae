clear;

files = dir('*.csv');
for i=1:length(files)
    eval(['load ' files(i).name ' -ascii']);
end
all_vectors = [X107m;X108m;X109m;X111m;X118m;X119m;
                    X124m;X200m;X203m;X207m;X208m;X210m;
                    X212m;X214m;X217m;X221m;X222m;X223m;
                    X228m;X230m;X231m;X232m;X233m];
abnormal_samples_std = all_vectors;

csvwrite('C:\repos\cae\data\conv1d\abnormal_samples_std.csv',abnormal_samples_std);