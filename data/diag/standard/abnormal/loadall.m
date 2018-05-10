clear;

files = dir('*.csv');
for i=1:length(files)
    eval(['load ' files(i).name ' -ascii']);
end
all_vectors = [X107m;X108m;X109m;X112m;X113m;X114m;
                    X115m;X116m;X119m;X200m;X201m;X202m;
                    X203m;X207m;X208m;X209m;X210m;X212m;
                    X213m;X214m;X217m;X219m;X221m;X222m;
                    X228m;X231m;X232m;X233m];
abnormal_samples_std = all_vectors;

csvwrite('C:\repos\cae\data\conv1d\abnormal_samples_std.csv',abnormal_samples_std);