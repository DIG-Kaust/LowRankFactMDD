class mdd_xlr(): 
    
    def __init__(self, d, ns, nr, nk): 
        self.d = d.reshape(ns, nr)
        self.lrshape = [(nr, nk), (nk, nr)]
        
    def forward(self, x): 
        ret = self.d
        for idx, item in enumerate(x):
            ret = ret.dot(item.reshape(self.lrshape[idx]))
        return ret.ravel()
    
    def adjoint(self, res, xx, idx): 
        
        res = res.reshape(self.d.shape)
        if 0 == idx:
            ret = self.d.conj().T.dot(res).dot(xx[1].reshape(self.lrshape[1]).conj().T)
        if 1 == idx:
            ret = self.d.dot(xx[0].reshape(self.lrshape[0])).conj().T.dot(res)  
        return ret.ravel()

class mdd_xqqt():
    
    def __init__(self, d, ns, nr, nk):
        self.d = d.reshape(ns, nr)
        self.qshape = [(nr, nk),]

    def forward(self, x): 
        q = x[0].reshape(self.qshape[0])
        ret = self.d.dot(q).dot(q.T)

        return ret.ravel()
    
    def adjoint(self, res, xx, idx): 
        
        res = res.reshape(self.d.shape)
        
        ret = self.d.conj().T.dot(res)
        ret = ret+ret.T
        ret = ret.dot(xx[0].reshape(self.qshape[0]).conj())
        
        return ret.ravel()



    
