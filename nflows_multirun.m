function nflows_multirun(parset_file, simset_file, Il, spc, kl)

rawdatafolder = strcat('C:\\Users\\Sebastian\\Documents\\MATLAB\\neural-flows\\TVBRawData\\', parset_file, simset_file);
coarsedatafolder = strcat('C:\\Users\\Sebastian\\Documents\\MATLAB\\neural-flows\\TVBData\\', parset_file, simset_file);
jsonfilefolder = strcat('C:\\Users\\Sebastian\\Documents\\MATLAB\\neural-flows\\TVBFiles\\', parset_file, simset_file);
savefigloc = strcat('C:\\Users\\Sebastian\\Documents\\MATLAB\\neural-flows\\OutputFigures\\', parset_file, simset_file);
outdatafolder = strcat('C:\\Users\\Sebastian\\Documents\\MATLAB\\neural-flows\\scratch\\', parset_file, simset_file);

files = dir(strcat(rawdatafolder, '\*.mat'));
Is = ["n1", "n075", "n05", "n025", "0", "025", "05", "075", "1"];
speeds = ["40", "60", "80", "100", "120", "140"];
ks = ["1_33", "2", "4" "24"];

if Il
for k = 1:length(files)
    fullpath = fullfile(files(k).folder, files(k).name);
    prep_data(fullpath, 1500, 1600, strcat(coarsedatafolder, '\\', simset_file, 'I', Is(k)), true, false)
end
end

if spc
for k = 1:length(files)
    fullpath = fullfile(files(k).folder, files(k).name);
    prep_data(fullpath, 1500, 1600, strcat(coarsedatafolder, '\\', simset_file, 'speed', speeds(k)), false, true)
end
end

if kl
for k = 1:length(files)
    fullpath = fullfile(files(k).folder, files(k).name);
    prep_data(fullpath, 1500, 1600, strcat(coarsedatafolder, '\\', simset_file, 'k', ks(k)), false, true)
end
end

files = dir(strcat(coarsedatafolder, '\*.mat'));

for k = 1:length(files)
    filename = files(k).name;
    filename = filename(1:length(filename)-4);
    pick_data(filename, parset_file, simset_file);
    template_nf(jsonfilefolder, filename, savefigloc);
end
end
    