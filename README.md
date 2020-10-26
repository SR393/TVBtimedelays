The role of heterogeneous time delays in large-scale brainwaves

The code in this repository was used to run simulations of brain activity in a brain network model, and to analyse various aspects of the model itself and the simulated activity. In particular, the project was undertaken to examine the role of time delays and time delay heterogeneity in a brain network model.
Two main subfolders are included in the repository, along with a folder for miscellaneous code. These main two subfolders are: 

1. Brainwaves_Code: this contains most of the code that was ultimately used in the project results.
2. Frequency-Delay Code: This contains code from investigations earlier in the project, which were ultimately excluded from the results.

Most code was written in python, with a small amount in Brainwaves_Code written in MATLAB in order to function with the neural-flows toolbox (https://github.com/brain-modelling-group/neural-flows). The simulations are run via neuroinformatics platfrom The Virtual Brain (TVB; https://github.com/the-virtual-brain/tvb-root).

Brainwaves_Code is split into 5 further folders, which are

i) Animations: scripts which create animations of simulated activity.

ii) IHCCs: scripts which extract signal phases, run cross correlations, and subsequently create interhemispheric cross-correlations for simulated data.

iii) R_local Curves: scripts which calculate local order parameters for set values of the radius of inclusion (OrderPars...) and then plot these to obtain R_local curves (rlocals...) for coherence analysis.

iv) Simulations: scripts which run simulations using TVB.

v) neural-flows analysis: scripts which take simulated data, cut 1500-1600ms segment, temporally coarse-grain, and then run it through neural-flows to obtain flow patterns.

Install and Version Information

Most code is designed with the sole purpose of functioning on the author's system as of the time of writing, so that in many cases direct references are made to paths specific to that purpose. All code was designed to run under the following specifications and language/toolbox versions.

OS:
  Windows 10, 64-bit

python Manager: 

miniconda. All packages installed and used are listed below. Some important packages and those referenced directly in the code are in bold.
  
  ###### Name: Version                  
  
  astroid: 2.4.2                   
  
  blas: 1.0                         
  
  ca-certificates: 2020.6.24                     
  
  certifi: 2020.6.20               
  
  colorama: 0.4.3                    
  
  cycler: 0.10.0                 
  
  decorator: 4.4.2              
  
  icc_rt: 2019.0.0             
  
  intel-openmp: 2020.1                  
  
  isort: 4.3.21                 
  
  joblib: 0.16.0                  
  
  kiwisolver: 1.2.0                
  
  lazy-object-proxy: 1.4.3           
  
  llvmlite: 0.33.0                
  
  **matplotlib: 3.3.0i**
  
  mccabe: 0.6.1          
  
  mkl: 2020.1          
  
  mkl-service: 2.3.0         
  
  mkl_fft: 1.1.0         
  
  mkl_random: 1.1.1         
  
  networkx: 2.4            
  
  numba: 0.50.1          
  
  numexpr: 2.7.1              
  
  **numpy: 1.19.1**
  
  numpy-base: 1.19.1
  
  openssl: 1.1.1g
  
  pillow: 7.2.0
  
  **pip: 20.2.1**
  
  pylint: 2.5.3
  
  pyparsing: 2.4.7
  
  **python: 3.8.5**
  
  python-dateutil: 2.8.1
  
  scikit-learn: 0.23.1
  
  **scipy: 1.5.0**
  
  setuptools: 49.2.1
  
  six: 1.15.0
  
  sqlite: 3.32.3
  
  threadpoolctl: 2.1.0
  
  toml: 0.10.1
  
  **tvb-data: 2.0**
  
  **tvb-library: 2.0.8**
  
  typing: 3.7.4.3
  
  vc: 14.1
  
  vs2015_runtime: 14.16.27012
  
  wheel: 0.34.2
  
  wincertstore: 0.2
  
  wrapt: 1.11.2
  
  zlib: 1.2.11

  Installation process: 
  1. Download miniconda (https://docs.conda.io/en/latest/miniconda.html)
  2. Install necessary packages (numpy, scipy, tvb-data, tvb-library) using the pip command 

  Necessary components of TVB are all containted in tvb-data and tvb-library excluding the network, which was later imported to tvb-data.

MATLAB: 
  v2019b
  neural-flows toolbox can be installed via download of the source code at https://github.com/brain-modelling-group/neural-flows and extracting the files to the MATLAB path. On Windows 10, the extracted folder may be named neural-flows-master; if this is the case, rename to neural-flows.

