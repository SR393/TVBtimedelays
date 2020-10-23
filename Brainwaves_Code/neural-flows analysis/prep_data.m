%transposes locs file if necessary, changes data filename, and
%runs through coarsgrain() to coarse-grain data nad cut from 1500ms - 1600ms
function prep_data(datafile, st, fn, outname, Il, spc)
if Il
load(datafile);
locs = locs.';
data = coarsegrain(data, 4, 0.0625, st, fn);
save(strcat(outname, '.mat'), 'data', 'locs')
end
if spc
load(datafile);
data = coarsegrain(data, 4, 0.0625, st, fn);
save(strcat(outname, '.mat'), 'data', 'locs')
end
end


