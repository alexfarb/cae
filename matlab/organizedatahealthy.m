function organizedatahealthy = organizedatahealthy(colposition) 
    folder = 'C:\repos\ann-arrhythmia\diag\normalized\healthy\';
    files = dir([folder '*.csv']);

    for i=1:length(files)
        load([folder files(i).name]);
    end

    all_vectors = [X100m;X101m;X103m;X105m;X106m;X111m;
                    X117m;X118m;X121m;X122m;X123m;X124m;
                    X205m;X215m;X220m;X223m;X230m;X234m;];
    
    all_transposed = all_vectors';
                
    xa = all_transposed(1:1200,colposition:colposition);
    xb = all_transposed(1201:2400,colposition:colposition);
    xc = all_transposed(2401:3600,colposition:colposition);
    organizedatahealthy = [xa xb xc];
end