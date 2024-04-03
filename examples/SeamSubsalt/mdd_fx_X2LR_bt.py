import sys 
sys.path.append('/home/chenf/mdd/src/')

import numpy as np

from scipy.fftpack import next_fast_len

from numpy import csingle, zeros, eye, sqrt, diag
from scipy.linalg import svdvals, svd 
from scipy.sparse.linalg import svds    

from numpy.random import rand
from numpy.linalg import norm

from ipalm_bt import ipalm_bt as ipalm
from mdd_XLR_utils import diffLR, gradLR_L, gradLR_R, prox_L1, prox_Fro

import argparse

from mpi4py import MPI 

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser(description="some parameters for mdd"    )   
group = parser.add_mutually_exclusive_group()

group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-q", "--quiet",   action="store_true")

parser.add_argument("-fstart",  type=int, help="the start of frequencies")
parser.add_argument("-fend",    type=int, help="the end of frequencies")
# type = lambda expression to make -nit=1e5 (float) valid
parser.add_argument("-nit", type=lambda x: int(float(x)), help="the number of iterations")

parser.add_argument("-tol", type=float, help="tolerance of relative msf change")

parser.add_argument("-inertia", type=float, help="inertia parameter")
parser.add_argument("-nr",   type=int, help="the rank of L and R")

parser.add_argument("-rgLR",  type=int, help="regularization methods for L and R, only two options")

# parser.add_argument("-lam", nargs="+", type=list, default=[])
parser.add_argument("-lam", nargs="+", type=str)
parser.add_argument("-lamNorm", action="store_true")

args = parser.parse_args()
# print(args.lam)

ilam = args.lam[0].split(",")
ilam = [float(iv) for iv in ilam]

# print(args.verbose)
gradlist=[gradLR_L, gradLR_R]

if 0 == args.rgLR:
    regType = 'BothF'
    proxlist=[prox_Fro, prox_Fro]

elif 1 == args.rgLR:
    regType = 'XlFXrL'
    proxlist=[prox_Fro, prox_L1]

else:
    raise TypeError("Regularization type unavailable")

# liplist=[lipL, lipR]

totalf =  args.fend - args.fstart ## fend excluded

freqsPerProcess = totalf // size

inc = [2.0, 2.0]

for ifreq in range(freqsPerProcess+1):

    idx_fname = min(args.fstart + ifreq + rank * freqsPerProcess, args.fend-1)

    ia = np.load('./input/normA_npad_0_fidx_%d.npy'%idx_fname)
    ib = np.load('./input/normB_npad_0_fidx_%d.npy'%idx_fname)

    # x0 = ia.T.conj().dot(ib)

    # x0 = x0/(norm(x0)*norm(ia))*norm(ib) 

    # u, s, vh = svd(x0)
    # s = sqrt(s[:args.nr])
    # u1 = u[:,:args.nr].dot(diag(s))
    # u2 = diag(s).dot(vh[:args.nr])

    # np.random.seed(5) 
    # u1 = rand(ia.shape[-1], args.nr) + 1j * rand(ia.shape[-1], args.nr) 
    # u2 = rand(args.nr, ia.shape[-1]) + 1j * rand(args.nr, ia.shape[-1]) 
    #u2 = rand(*u2.shape) + 1j * rand(*u2.shape) 

    u1 =  1j * zeros((ia.shape[-1], args.nr))
    u1[:args.nr, :args.nr] = eye(args.nr) + eye(args.nr) * 1j
    # x0 = [u1] 
    x0 = [u1, u1.T]

    tau_guess = [norm(ia, 'fro')**2, norm(ia, 'fro')**2]
    x1 = ipalm(gradlist, proxlist, tau_guess, inc, diffLR, args, ia, ib, x0)

    x1 = x1[0].dot(x1[1])

    np.save('./output/XLR_%d_%s_r_%d_lam1_%.1f_lam2_%.1f_lamNorm_%d_rinit_inertia_%.1f_bt'%(idx_fname, regType, args.nr, ilam[0], ilam[1], args.lamNorm, args.inertia), x1)
