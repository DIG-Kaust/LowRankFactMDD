import sys
sys.path.append('/home/chenf/mdd/src/')
from scipy.ndimage import gaussian_filter
import numpy as np

from scipy.fftpack import next_fast_len

from numpy import csingle, zeros, eye, sqrt, diag
from scipy.linalg import svdvals, svd
from scipy.sparse.linalg import svds    

from numpy.random import rand
from numpy.linalg import norm

from ipalm_bt import ipalm_bt as ipalm
from mdd_XLR_utils import prox_L1, prox_Fro, diffQQt, gradQQt

import argparse

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

##########################################################################################

parser = argparse.ArgumentParser(description="some parameters for mdd"    )
group = parser.add_mutually_exclusive_group()

group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-q", "--quiet",   action="store_true")

parser.add_argument("-fstart",  type=lambda x: int(float(x)), help="the start of frequencies")
parser.add_argument("-fend",    type=lambda x: int(float(x)), help="the end of frequencies")
    
parser.add_argument("-nit", type=lambda x: int(float(x)), help="the number of iterations")
    
parser.add_argument("-tol", type=float, help="tolerance of relative msf change")
parser.add_argument("-inertia", type=float, help="inertia parameter")
    
parser.add_argument("-nr", type=int, help="the rank of L and R")

parser.add_argument("-rgQ",  type=int, help="regularization methods for Q")

parser.add_argument("-lam", nargs="+", help="regularization parameter for Q")

parser.add_argument("-lamNorm", action="store_true")

args = parser.parse_args()

##########################################################################################

gradlist=[gradQQt]

inc = [2.0]

if 0 == args.rgQ:

    proxlist=[prox_Fro]
    regType = 'QF'
elif 1 == args.rgQ:
    regType = 'QL1'
    proxlist=[prox_L1]

else:
    raise TypeError("Regularization type unavailable")

# liplist=[lipQ]

totalf =  args.fend - args.fstart ## fend excluded

freqsPerProcess = totalf // size

for ifreq in range(freqsPerProcess+1):

    idx_fname = min(args.fstart + ifreq + rank * freqsPerProcess, args.fend-1)

    ia = np.load('./input/A_npad_0_fidx_%d.npy'%idx_fname)
    ib = np.load('./input/B_npad_0_fidx_%d.npy'%idx_fname)

    #x0 = ia.T.conj().dot(ib)
    u1 =  1j * zeros((ia.shape[-1], args.nr))
    u1[:args.nr, :args.nr] = eye(args.nr) + eye(args.nr) * 1j
    x0 = [u1]
    # print(x0[0].shape) 
    tau_guess = [norm(ia, 'fro')**2]

    x1 = ipalm(gradlist, proxlist, tau_guess, inc, diffQQt, args, ia, ib, x0)

    # x1 = x1[0].dot(x1[0].T)

    #print(type(x1), type(args.lam[0]))

    np.save('./output/XQQt_warm_init_%d_%s_r_%d_lam_%s_lamNorm_%d_rinit_inertia_%.1f_bt'%(idx_fname, regType, args.nr, args.lam[0], args.lamNorm, args.inertia), x1[0])
