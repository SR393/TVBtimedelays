The role of heterogeneous time delays in large-scale brainwaves

The code in this repository was used to run simulations of brain activity in a brain network model, and to analyse various aspects of the model itself and the simulated activity. In particular, the project was undertaken to examine the role of time delays and time delay heterogeneity in a brain network model.
Three main subfolders are included in the repository, along with a folder for miscellaneous code. These main three subfolders are: 

1. Brainwaves_Code: this contains most of the code that was ultimately used in the project results.
2. IHCCs: this contains the remaining code used in the project results, specifically for evaluating interhemispheric cross-correlations.
3. Frequency-Delay Code: This contains code from investigations earlier in the project, which were ultimately excluded from the results.

Most code was written in python, with a small amount in Brainwaves_Code written in MATLAB in order to function with the neural-flows toolbox (https://github.com/brain-modelling-group/neural-flows). The simulations are run via neuroinformatics platfrom The Virtual Brain (TVB; https://github.com/the-virtual-brain/tvb-root). Most code is designed with the sole purpose of functioning on the author's system as of the time of writing, so that in many cases direct references are made to paths specific to that purpose.
