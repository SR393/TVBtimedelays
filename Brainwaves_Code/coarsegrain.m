#coarse-grains data (averages over every 4 data points) and cuts from 1500 - 1600ms
function out = coarsegrain(data, avg_period, dt, st, fin)
st = st/dt + 1;
fin = fin/dt;
data = data(st:fin, :);
coarsedata = zeros(length(data)/avg_period, 512);
for i = 1:512
    data_i = buffer(data(:, i), avg_period);
    coarsedata(:, i) = mean(data_i);
end
out = coarsedata;
end


    
