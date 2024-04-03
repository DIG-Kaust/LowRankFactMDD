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

mpiexec -n 16 python3 mdd_fx_X2QQt_bt.py -fstart=10 -fend=599 -nit=200000 -tol=1e-7 -nr=25 -lam=0.01 -rgQ=0 -inertia=0.0 # -alpha=1e-2 -eta=2 

## mpiexec -n 16 python3 mdd_fx_X2LR_bt.py -fstart=10 -fend=591 -nit=200000 -tol=1e-7 -n    r=25 -lam=1,1 -rgLR=0 -inertia=0.0
                                                  
