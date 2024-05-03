import numpy as np

class alpha_type1:
    t0 = 0
    def __init__(self):
        self.t0 = alpha_type1.t0
    def pop(self):
        alpha = self.t0/(self.t0+3)
        self.t0 += 1
        return alpha

class alpha_type2:
    t0 = 1
    def __init__(self):
        self.t0 = alpha_type2.t0
    def pop(self):
        t1 = 0.5*(1+np.sqrt(1+4*self.t0**2))
        alpha = (self.t0-1.0) / t1
        self.t0 = t1
        return alpha

def ipalm(A, b, prox, tau, alpha, tol=1e-5, scale=[0.5, 0.5], lam=[0.1,0.1], iter_lim=None, show=False, x0=None, beta=None):

    gnorm0 = np.inf

    x1 = list(x0)
    x2 = list(x0)

    if show:
            print("%5s%20s%20s%20s\n"%("niter", "data_residual", "norm_grad", "rel_grad_diff"))

    for it in range(0, iter_lim):

        gnorm = 0

        for iblck in range(len(x0)):

            yiblck = x1[iblck] + alpha.pop() * (x1[iblck] - x0[iblck])

            if beta is not None:
                ziblck = x1[iblck] + beta * (x1[iblck] - x0[iblck])
            else:
                ziblck = yiblck

            x2[iblck] = ziblck

            res = A.forward(x2) - b
            f0 = 0.5*np.linalg.norm(res)**2
            grad = A.adjoint(res, ziblck, iblck)
            gnorm += np.linalg.norm(grad)

            while(True):

                x1_tmp = yiblck-tau[iblck]*grad
                x1_tmp = prox[iblck](x1_tmp, lam[iblck]*tau[iblck])
                x2[iblck] = x1_tmp

                res = A.forward(x2) - b
                f1 = 0.5*np.linalg.norm(res)**2

                if backtrack(f1-f0, x1_tmp-yiblck, grad, tau[iblck]) or tau[iblck]<1e-12:
                    break
                else:
                    tau[iblck] *= scale[iblck]

        x0 = x1
        x1 = x2

        if show:
            print("%5d%20.3f%20.3f%20.6f"%(it, np.linalg.norm(res), np.linalg.norm(grad), np.abs(gnorm/gnorm0-1.0)))

        if np.abs(gnorm/gnorm0-1.0)<tol:
            break

        gnorm0 = gnorm

    return x2

def l2_prox(v, t):

    return v/(2*t+1)


def backtrack(diff_misfit, diff_model, gv, tau):

    # https://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/08-prox-grad.pdf

    if diff_misfit < gv.dot(diff_model).real + 0.5*diff_model.dot(diff_model)/tau:
        return True
    else:
        return False

