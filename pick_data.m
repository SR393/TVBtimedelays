function pick_data(filename, parset_file, parfolder)
fid = fopen('C:\\Users\\Sebastian\\Documents\\MATLAB\\neural-flows\\TVBFiles\\template.json', 'r');
i = 1;
tline = fgetl(fid);
A{i} = tline;
while ischar(tline)
    i = i+1;
    tline = fgetl(fid);
    A{i} = tline;
end
fclose(fid);

A{5} = sprintf('%s', strcat('      "dir": "C:\\Users\\Sebastian\\Documents\\MATLAB\\neural-flows\\scratch\\', parset_file, parfolder, '",'));
A{6} = sprintf('%s', strcat('      "dir_tmp":','"/', filename, '",'));
A{9} = sprintf('%s', strcat('          "filename":', ' "', filename, '.json"', ','));
A{10} = sprintf('%s', strcat('          "dir": "C:\\Users\\Sebastian\\Documents\\MATLAB\\neural-flows\\TVBFiles\\', parset_file, parfolder, '"')); 
A{13} = sprintf('%s', strcat('          "filename":', ' "', filename, '_out.json"', ','));
A{14} = sprintf('%s', strcat('          "dir": "C:\\Users\\Sebastian\\Documents\\MATLAB\\neural-flows\\TVBFiles\\', parset_file, parfolder, '"')); 
A{30} = sprintf('%s', strcat('      "label":', ' "', filename, '_interp",'));
A{53} = sprintf('%s', strcat('      "label":', ' "', filename, '_flows"'));
A{150} = sprintf('%s', strcat('          "dir": "C:\\Users\\Sebastian\\Documents\\MATLAB\\neural-flows\\TVBData\\', parset_file, parfolder, '",')); 
A{151} = sprintf('%s', strcat('       "name":', ' "', filename, '.mat"'));

fid = fopen(strcat('C:\\Users\\Sebastian\\Documents\\MATLAB\\neural-flows\\TVBFiles\\', parset_file, parfolder, '\\', filename, '.json'), 'w');
for i = 1:numel(A)
    if A{i+1} == -1
        fprintf(fid,'%s', A{i});
        break
    else
        fprintf(fid,'%s\n', A{i});
    end
end
