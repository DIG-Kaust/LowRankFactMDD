#!/bin/bash
##SBATCH --nodes 2
#SBATCH --partition=batch
#SBATCH -J mdd_xlr
#SBATCH -o mdd_xlr.%J.out
#SBATCH --time=620:00
#SBATCH --mem=50G

#SBATCH --ntasks=16
#SBATCH --cpus-per-task=16

#SBATCH --constraint=amd
module loading python/3.7.0
module loading mpi4py/3.0.3-py3.7.0-ompi4.0.3

# module loading anaconda3/2019.10

mpiexec -n 16 python3 mdd_fx_X2QQt_bt_warm_init.py -fstart=0 -fend=430 -nit=200000 -tol=1e-7 -nr=18 -lam=0.1 -rgQ=0 -inertia=0.0 ###-alpha=1e-2 -eta=2 
                               
