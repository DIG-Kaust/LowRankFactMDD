from numpy import csingle, zeros, hstack, inf, sum, abs, log 
from numpy.linalg import norm
from scipy.linalg import svdvals
from scipy.sparse.linalg import svds    

def ipalm_bt(grads, proxs, tau_guess, inc, diff, argv, ia, ib, xs0): 
    
    nblocks=len(grads)
    # initialize states for inertia
    xs1=list(xs0)
    xs2=list(xs0)
    
    it=0 #iteration counter
    
    lam = argv.lam[0].split(",") 
    lam = [float(iv) for iv in lam]
    
    if argv.lamNorm:
        lam = [iv*log(1.0+norm(ia.T.conj().dot(ib),ord='fro')) for iv in lam]    


    # print(lam)
    nv, nnv = zeros(nblocks), zeros(nblocks)
    #main loop
    total_msf_old = inf 
    # data_msf_old = inf
    BREAK = 0 
    #tau_tmp = tau_guess[i]
    while it < argv.nit: # and err > tol: 
        reg_msf = 0.0    
        #compute inertial coefficients
        if 0 == argv.inertia: 
            inertia = it/(it+3.0)
        else:
            inertia = argv.inertia
    
        for i in range(nblocks):

            yi = xs1[i] + inertia * (xs1[i] - xs0[i])
            zi = xs1[i] + inertia * (xs1[i] - xs0[i])
    
            xs2[i] = zi
    
            # Estimate the resuial and gradient with the current x
            res = diff(xs2, ia, ib) 
            data_msf_cur = 0.5*norm(res, ord='fro')**2 
            gg = grads[i](xs2, res, ia) 
    
            ############################################################## 

            cnt = 0 
            while(True):    
    
                tau = tau_guess[i] * inc[i]**cnt
                # print(cnt, tau) 
                y_tmp = yi - 1.0/tau * gg
    
                # x_new, nv[i], nnv[i] = proxs[i](y_tmp, tau, lam[i])
                x_new, nv[i], nnv[i] = proxs[i](y_tmp, tau, lam[i])
                xs2[i] = x_new
                res = diff(xs2, ia, ib) 

                data_msf_new = 0.5*norm(res, ord='fro')**2 
                # F(P(y_k))<Q(P(y_k), y_k)
                if data_msf_new < data_msf_cur + 0.5 * ( norm(x_new - y_tmp, ord='fro')**2 * tau - norm(gg, ord='fro')**2 / tau ):# or cnt > 20: ## debug ? cnt>20
                    break
                cnt += 1
    
            # tau_guess[i] = tau
            reg_msf += nv[i]*lam[i]/tau

        total_msf_cur = reg_msf + data_msf_cur
    
        if argv.verbose:
            print("%15.3f%18d"%(total_msf_cur,cnt+1), end='', flush=True)
            for i in range(nblocks):
                print("%15.3f%15.3f"%(nv[i], nnv[i]), end='', flush=True)
            print("\n")
        #print(argv.tol)
        #if abs(data_msf_new-data_msf_cur) < argv.tol:
        # if abs(total_msf_cur/total_msf_old-1.0) < argv.tol or total_msf_cur > total_msf_old:
        if abs(total_msf_cur/total_msf_old-1.0) < argv.tol:
        #if abs(data_msf/data_msf_old-1.0) < argv.tol:
        # if abs(total_msf/total_msf_old-1.0) < argv.tol or total_msf>total_msf_old:
            # print(it)
            break
    
        total_msf_old = total_msf_cur
        # total_msf_old = total_msf     
        
        xs0 = list(xs1)
        xs1 = list(xs2)
        it+=1 
    
    
    return xs1 
