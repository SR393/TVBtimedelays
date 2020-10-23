%starts process of running simulation data through neural-flows
%by choosing parameter set (K-s, lambda-k, or lambda-I) and running
%nflows_analyse()
%inputs are Booleans, indicating which parameter set to run through - only make one true, others false
function nflows_lambdas(Il, spc, kl)
filestrings_lambdas = ["lambda50", "lambda100", "lambda150", "lambda200", "lambda250", "lambda300"];
filestrings_couplings = ["coupling05", "coupling06", "coupling07", "coupling08"];
if Il
    for k = 1:length(filestrings_lambdas)
        nflows_analyse('k-lambda\\', filestrings_lambdas(k), Il, spc, kl)
    end
end
if spc
    for k = 1:length(filestrings_couplings)
        nflows_analyse('speed-coupling\\', filestrings_couplings(k), Il, spc, kl)
    end
end
if kl
    for k = 1:length(filestrings_lambdas)
        nflows_analyse('k-lambda-rand\\', filestrings_lambdas(k), Il, spc, kl)
    end
end
end
