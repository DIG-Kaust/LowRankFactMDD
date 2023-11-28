from numpy.linalg import norm
from numpy import zeros, abs, sign, maximum, diag, array, where

from scipy.sparse.linalg import svds

def diffQQt(xs, ia, ib):
    xs = xs[0]
    r = ia.dot(xs@xs.T) - ib
    return r

def gradQQt(xs, ir, ia):

    xs = xs[0]
    AhR = ia.T.conj().dot(ir)

    # ret1 = ia.T.conj().dot(r).dot(xr.T.conj())
    # ret2 = xl.T.conj().dot( ia.T.conj().dot(r) )
    print(xs.shape)
    dd = (AhR +  AhR.T).dot(xs.conj())

    return dd # , 0.5*norm(rr, ord='fro')**2 


def diffLR(xs, ia, ib):

    xl, xr = xs[0], xs[1]

    r = ia.dot(xl@xr) - ib

    # data_msf = 0.5*norm(r, ord='fro')**2 

    return r # data_msf

def gradLR_L(xs, ir, ia):

    xl, xr = xs[0], xs[1]

    ret = ia.T.conj().dot(ir).dot(xr.T.conj())

    #data_msf = 0.5*norm(r, ord='fro')**2 

    return ret#, data_msf


def gradLR_R(xs, ir, ia):

    xl, xr = xs[0], xs[1]

    # r = ia.dot(xl@xr) - ib

    ret = xl.T.conj().dot( ia.T.conj().dot(ir) )

    # data_msf = 0.5*norm(r, ord='fro')**2 

    return ret# , data_msf

def diffQQt_Xtile(xs, ia, ib):

    # num_xTile, mTile, kTile = xs[0]
    # xs = xs[1]
    # print(xs.shape)  

    # num_xTile number of receiver lines
    # mTile     number of receivers each line
    num_xTile, mTile = xs[0]
    xs = xs[1]
    # print(mTile, num_xTile)
    x = zeros((num_xTile*mTile, num_xTile*mTile), dtype = xs[0].dtype)

    for i in range(num_xTile):
        for j in range(i+1):

            ixs = xs[i*(i+1)//2+j]

            if i==j:
                ixs = ixs.dot(ixs.T)
                x[i*mTile:(i+1)*mTile, i*mTile:(i+1)*mTile] = ixs
            else:
                ixs_l, ixs_r = ixs[0], ixs[1]
                ixs = ixs_l.dot(ixs_r)
                x[i*mTile:(i+1)*mTile, j*mTile:(j+1)*mTile] = ixs
                x[j*mTile:(j+1)*mTile, i*mTile:(i+1)*mTile] = ixs.T

    r = ia.dot(x) - ib
    # print(x)
    return r,x

def gradQQ_Xtile(xs, ir, ia):

    num_xTile, mTile = xs[0]
    xs = xs[1]

    AhR = ia.T.conj().dot(ir)

    # print(xs.shape) 
    # dd = (AhR +  AhR.T).dot(xs[1][0].conj())   
    AhR_ = (AhR +  AhR.T) # .dot(xs[1][0].conj())   
    dd = []
    for i in range(num_xTile):
        for j in range(i+1):
            ixs = xs[i*(i+1)//2+j]
            if i==j:
                ixs = AhR_[i*mTile:(i+1)*mTile, i*mTile:(i+1)*mTile].dot(ixs.conj())
                dd.append(ixs)
            else:
                # ixs_l, ixs_r = ixs[0], ixs[1]
                dXL = (AhR[i*mTile:(i+1)*mTile, j*mTile:(j+1)*mTile]+AhR[j*mTile:(j+1)*mTile, i*mTile:(i+1)*mTile].T).dot(ixs[-1].T.conj())
                dXR = ixs[0].T.conj().dot(AhR[i*mTile:(i+1)*mTile, j*mTile:(j+1)*mTile]+AhR[j*mTile:(j+1)*mTile, i*mTile:(i+1)*mTile].T)
                dd.append([dXL, dXR])
    return dd
