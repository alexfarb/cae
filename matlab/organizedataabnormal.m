function organizedata = organizedata(colposition) 
    folder = 'C:\repos\ann-arrhythmia\diag\standard\abnormal\';
    files = dir([folder '*.csv']);

    for i=1:length(files)
        load([folder files(i).name]);
    end

    all_vectors = [X107m;X108m;X109m;X112m;X113m;X114m;
                    X115m;X116m;X119m;X200m;X201m;X202m;
                    X203m;X207m;X208m;X209m;X210m;X212m;
                    X213m;X214m;X217m;X219m;X221m;X222m;
                    X228m;X231m;X232m;X233m];
    
    all_transposed = all_vectors';
                
    xa = all_transposed(1:1200,colposition:colposition);
    xb = all_transposed(1201:2400,colposition:colposition);
    xc = all_transposed(2401:3600,colposition:colposition);
    organizedata = [xa xb xc];
end