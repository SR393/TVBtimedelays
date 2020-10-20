function nflows_lambdas(Il, spc, kl)
filestrings_lambdas = ["lambda50", "lambda100", "lambda150", "lambda200", "lambda250", "lambda300"];
filestrings_couplings = ["coupling07", "coupling08"];
if Il
    for k = 1:length(filestrings_lambdas)
        nflows_multirun('k-lambda\\', filestrings_lambdas(k), Il, spc, kl)
    end
end
if spc
    for k = 1:length(filestrings_couplings)
        nflows_multirun('speed-coupling\\', filestrings_couplings(k), Il, spc, kl)
    end
end
if kl
    for k = 1:length(filestrings_lambdas)
        nflows_multirun('k-lambda-rand\\', filestrings_lambdas(k), Il, spc, kl)
    end
end
end