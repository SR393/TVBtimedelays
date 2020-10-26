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
  Name                    Version                   Build  Channel
  astroid                   2.4.2                    py38_0
  blas                      1.0                         mkl
  ca-certificates           2020.6.24                     0
  certifi                   2020.6.20                py38_0
  colorama                  0.4.3                      py_0
  cycler                    0.10.0                   pypi_0    pypi
  decorator                 4.4.2                    pypi_0    pypi
  icc_rt                    2019.0.0             h0cc432a_1
  intel-openmp              2020.1                      216
  isort                     4.3.21                   py38_0
  joblib                    0.16.0                     py_0
  kiwisolver                1.2.0                    pypi_0    pypi
  lazy-object-proxy         1.4.3            py38he774522_0
  llvmlite                  0.33.0                   pypi_0    pypi
  **matplotlib                3.3.0                    pypi_0    pypi**
  mccabe                    0.6.1                    py38_1
  mkl                       2020.1                      216
  mkl-service               2.3.0            py38hb782905_0
  mkl_fft                   1.1.0            py38h45dec08_0
  mkl_random                1.1.1            py38h47e9c7a_0
  networkx                  2.4                      pypi_0    pypi
  numba                     0.50.1                   pypi_0    pypi
  numexpr                   2.7.1                    pypi_0    pypi
  **numpy                     1.19.1           py38h5510c5b_0**
  numpy-base                1.19.1           py38ha3acd2a_0
  openssl                   1.1.1g               he774522_1
  pillow                    7.2.0                    pypi_0    pypi
  **pip                       20.2.1                   py38_0**
  pylint                    2.5.3                    py38_0
  pyparsing                 2.4.7                    pypi_0    pypi
  **python                    3.8.5                he1778fa_0**
  python-dateutil           2.8.1                    pypi_0    pypi
  scikit-learn              0.23.1           py38h25d0782_0
  **scipy                     1.5.0            py38h9439919_0**
  setuptools                49.2.1                   py38_0
  six                       1.15.0                     py_0
  sqlite                    3.32.3               h2a8f88b_0
  threadpoolctl             2.1.0              pyh5ca1d4c_0
  toml                      0.10.1                     py_0
  **tvb-data                  2.0                      pypi_0    pypi**
  **tvb-library               2.0.8                    pypi_0    pypi**
  typing                    3.7.4.3                  pypi_0    pypi
  vc                        14.1                 h0510ff6_4
  vs2015_runtime            14.16.27012          hf0eaf9b_3
  wheel                     0.34.2                   py38_0
  wincertstore              0.2                      py38_0
  wrapt                     1.11.2           py38he774522_0
  zlib                      1.2.11               h62dcd97_4

  Installation process: 
  1. Download miniconda (https://docs.conda.io/en/latest/miniconda.html)
  2. Install necessary packages (numpy, scipy, tvb-data, tvb-library) using the pip command 

  Necessary components of TVB are all containted in tvb-data and tvb-library excluding the network, which was later imported to tvb-data.

MATLAB: 
  v2019b
  neural-flows toolbox can be installed via download of the source code at https://github.com/brain-modelling-group/neural-flows and extracting the files to the MATLAB path. On Windows 10, the extracted folder may be named neural-flows-master; if this is the case, rename to neural-flows.

