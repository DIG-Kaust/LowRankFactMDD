<!---## ![LOGO](https://github.com/DIG-Kaust/Project_Template/blob/master/logo.png)--->

Reproducible material for **A physics-aware, low-rank regularization for multidimensional deconvolution - Fuqiang Chen, Matteo Ravasi, and David Keyes** submitted to Geophysics.

## Project structure
This repository is organized as follows:

* :open_file_folder: **mdd_lrr**: python library containing routines to perform mdd with factorization-based low-rank regularization;

**NOTE**: due to their large size, the datasets used in this repository cannot be shared directly in this repository. 
Refer to the ``README`` file in the ``data`` folder for more details.

## Notebooks
The following notebooks are provided:

* **:open_file_folder: seam1**:

  - :orange_book: ``mdd_lrr_toy.ipynb``: notebook performing mdd with random complex matrices for down- and up-going wavefield;
    
## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

Remember to always activate the environment by typing:
```
conda activate pocsdeblend
```

<!---**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.--->
