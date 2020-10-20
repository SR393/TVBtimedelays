#Fits Weibull distribution to fibre length data, including
#intrahemispheric (right and left), interhemispheric subgroups

from tvb.simulator.lab import *
from tvb.datatypes import time_series
from tvb.basic.config import settings
import matplotlib.pyplot as plt
import os as os
import numpy as np
import scipy.stats as spst

def rounddec(x, n):
    x = np.rint(x*10**n)/10**n
    return x
    
folder = r'C:\Users\sebra\miniconda3\envs\TVBS2\FibreLengthHists'

#Load connectivity data for tract lengths
fivetwelve_node = 'geombinsurr_partial_000_Gs.zip'
network = fivetwelve_node
wm = connectivity.Connectivity.from_file(network)
tract_lens = wm.tract_lengths
numnodes = tract_lens.shape[0]

#Get data from matrices
tract_data = tract_lens.copy().flatten()
tract_data = tract_data[tract_data != 0]

left_intra = tract_lens[0:256, 0:256]
right_to_left = tract_lens[0:256, 256:512]
left_to_right = tract_lens[256:512, 0:256]
right_intra = tract_lens[256:512, 256:512]

left_intra = left_intra.flatten()
left_intra = left_intra[left_intra != 0]
right_intra = right_intra.flatten()
right_intra = right_intra[right_intra != 0]
right_to_left = right_to_left.flatten()
left_to_right = left_to_right.flatten()
inter = np.concatenate((right_to_left, left_to_right))

#Find statistical moments
left_mu = np.mean(left_intra)
right_mu = np.mean(right_intra)
inter_mu = np.mean(inter)

left_var = np.std(left_intra)**2
right_var = np.std(right_intra)**2
inter_var = np.std(inter)**2

#Set up for plots
x = np.linspace(0, 300, 1000)
k = spst.weibull_min.fit(tract_data, floc = 0)
pdf_weibull = spst.weibull_min.pdf(x, c = k[0], loc = 0, scale = k[2])

#Fit weibull dist. to data by finding parameters and make pdf line
leftintra_fit = spst.weibull_min.fit(left_intra, floc = 0) #fit
leftintra_pdf = spst.weibull_min.pdf(x, c = leftintra_fit[0], loc = 0, scale = leftintra_fit[2]) #PDF
rightintra_fit = spst.weibull_min.fit(right_intra, floc = 0)
rightintra_pdf = spst.weibull_min.pdf(x, c = rightintra_fit[0], loc = 0, scale = rightintra_fit[2])
inter_fit_weibull = spst.weibull_min.fit(inter, floc = 0)
inter_pdf_weibull= spst.weibull_min.pdf(x, c = inter_fit_weibull[0], loc = 0, scale = inter_fit_weibull[2])

bins = 50

#Plot whole brain data
fig = plt.figure(figsize = (14, 14))
plt.hist(tract_data, bins = 90, density = True)
plt.xlabel('Tract Lengths (mm)', size = 18)
plt.ylabel('Normalised Counts', size = 18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.text(0.01, 0.9, 'k = '+str(rounddec(k[0], 2))+'\n'+r'$\lambda$ = '+str(rounddec(k[2], 2)), transform=plt.transAxes)
plt.title('Histogram of Fibre Tract Lengths', size = 22)
plt.show()

#Composite plot (Intrahemispheric, interhemispheric, whole brain)
fig, axs = plt.subplots(2, 2, figsize = (12, 12))
axs[0, 0].hist(left_intra, bins = bins, density = True)
axs[0, 0].plot(x, rightintra_pdf)
axs[0, 0].set_ylabel('Probability Density/Normed Counts', size = 15)
axs[0, 0].set_title('Left Hemisphere', size = 17)
axs[0, 0].text(0.01, 0.9, ' k = '+str(rounddec(leftintra_fit[0], 2))+'\n'+r'$\lambda$ = '+str(rounddec(leftintra_fit[2], 2)), transform=axs[0, 0].transAxes)
axs[0, 1].hist(right_intra, bins = bins, density = True)
axs[0, 1].plot(x, rightintra_pdf)
axs[0, 1].set_title('Right Hemisphere', size = 17)
axs[0, 1].text(0.01, 0.9, ' k = '+str(rounddec(rightintra_fit[0], 2))+'\n'+r'$\lambda$ = '+str(rounddec(rightintra_fit[2], 2)), transform=axs[0, 1].transAxes)
axs[1, 0].hist(inter, bins = bins, density = True)
axs[1, 0].plot(x, inter_pdf_weibull, label = 'Weibull')
axs[1, 0].set_xlabel('Tract Lengths (mm)', size = 15)
axs[1, 0].set_ylabel('Probability Density/Normed Counts', size = 15)
axs[1, 0].set_title('Interhemispheric Connections', size = 17)
axs[1, 0].text(0.01, 0.9, 'k = '+str(rounddec(inter_fit_weibull[0], 2))+'\n'+r'$\lambda$ = '+str(rounddec(inter_fit_weibull[2], 2)), transform=axs[1, 0].transAxes)
axs[1, 1].hist(tract_data, bins = 50, density = True)
axs[1, 1].plot(x, pdf_weibull, label = 'Weibull')
axs[1, 1].set_xlabel('Tract Lengths (mm)', size = 15)
axs[1, 1].text(0.01, 0.9, 'k = '+str(rounddec(k[0], 2))+'\n'+r'$\lambda$ = '+str(rounddec(k[2], 2)), transform=axs[1, 1].transAxes)
axs[1, 1].set_title('Whole Brain', size = 17)
plt.suptitle('Fibre Length Histograms and Weibull Distribution Fits', size = 20)
plt.show()



