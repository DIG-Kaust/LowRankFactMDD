![LOGO](https://github.com/DIG-Kaust/Project_Template/blob/master/logo.png)

Reproducible material for **Seismic deblending with a hard data constraint - Ravasi M.** submitted to IMAGE 2024.

## Project structure
This repository is organized as follows:

* :open_file_folder: **pocsdeblend**: python library containing routines to perform deblending with various proximal solvers;
* :open_file_folder: **data**: folder where input data must be placed;
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);

**NOTE**: due to their large size, the datasets used in this repository cannot be shared directly in this repository. 
Refer to the ``README`` file in the ``data`` folder for more details.

## Notebooks
The following notebooks are provided:

* **:open_file_folder: seam1**:

  - :orange_book: ``Deblending_Seam1OBC.ipynb``: notebook performing deblending with different solvers and frequency-independent thresholding;
  - :orange_book: ``Deblending_Seam1OBC_freqthresh.ipynb``: notebook performing deblending with different solvers and frequency-dependent thresholding;
  - :orange_book: ``Deblending_Seam1OBC_visualization.ipynb``: notebook used to create the final figures in the paper.

* **:open_file_folder: seam1**:

  - :orange_book: ``3DBlending_FKIHT.ipynb``: notebook performing deblending with IHT solver;
  - :orange_book: ``3DBlending_FKHQS.ipynb``: notebook performing deblending with HQS solver;
  - :orange_book: ``3DBlending_FKADMM.ipynb``: notebook performing deblending with ADMM solver;
  - :orange_book: ``3DBlending_Visualization.ipynb``: notebook used to create the final figures in the paper.


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

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.
