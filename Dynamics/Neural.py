import numpy as np
from Dynamics.Net import Net 

class Neural(Net): 

    def __init__(self, N, p, name, topo, m, seed):
        Net.__init__(self, N, p, 'wc', name, topo,m, seed=seed)
        self.gt = np.array([1,1])
   
    def f(self, x, param ):
        return -x 
     
    def g(self, x, y, param ):
        tau, mu = param   
        return np.ones_like(x) / (1 + np.exp(- tau * (y - mu))) 
 
    def dFdx(self, x, t, param, A=None):
        '''
        g2 denotes the derivative of g w.r.t. the second argument 
        '''
        A = self.A
        N = self.N
        tau, mu = param   
        exp_term = np.exp(- tau * (x - mu))

        g2 = exp_term * tau / (1 + exp_term) ** 2

        Y = np.repeat(g2, N).reshape(N,N)
        Y = A * (Y.T)
        diag = np.full(N,-1)
        np.fill_diagonal(Y, diag)
        return Y
     
     
    def dxdt(self, x,t, param, A=None ): 
        if A is None: 
            A = self.A 
        return self.f(x, param) + A @ self.g(x, x, param)
  
    def perturb_err_matrix(self, param, x):
        Y = self.dFdx(x, 0, param)
        c = self.dxdt(x, 0, param)
        return Y, c
    
    def dxdt_mfa(self, x,t, param, degree):
        '''
        Return the N-D mean-field formula f(x, param) + degree * g(x, xeff, param).
        ''' 
        self.nfe_mfa += 1 
        xeff = self.xeff  
        return self.f(x,param) + degree * self.g(x,xeff,param)
    
    def dxdt_xeff(self, x,t, param, beta=None):
        '''
        Return the 1-D mean-field formula f(x, param) + beta * g(x, x, param).
        '''
        return self.f(x, param) + self.g(x,x,param) * beta
    
   