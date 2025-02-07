import numpy as np
from Dynamics.Net import Net
import scipy.sparse as sp

class Epidemic(Net):
    '''
    Create a set of methods for Ecology dynamics.
    '''
    def __init__(self, N=10, p=3, name='', topo='er', m=4, seed=42):
        Net.__init__(self, N, p, 'epi', name, topo, m, seed=seed)
        self.gt = np.array([0.5])

    def f(self, x): 
        B = 1
        return - B * x

    def g(self, x, y):
        R = self.unload(self.gt)[0]
        return R * (1 - x) * y

    def dFdx(self, x, t, param, A=None):
        A = self.A
        N = self.N
        R = param[0]
        B = 1
        Z = np.repeat(x, N).reshape((N, N))
        Y = A * (R * (1 - Z))
        diag = - B - R * (A @ x)
        np.fill_diagonal(Y, diag)
        return Y
     
    def dxdt_err(self, err, t, param, x):
        '''
        ==== Background ====
        Ref:  perturbation()

        Let e_i = x_i - z_i 

        d e_i / dt ~ [f'(z_i) + sum_j A_ij g_1 (z_i, z_j)] e_i + sum_j A_ij g_2 (z_i, z_j) e_j + f(z_i) + sum_j A_ij g(z_i, z_j) 
        where g1 (g2) is the partial derivative of g w.r.t. the first (second) argument
        
        Let a_i = [f'(z_i) + sum_j A_ij g_1 (z_i, z_j)]
        Let b_ij = g_2 (z_i, z_j)
        Let Y_ij = A_ij * b_ij if i != j; B_ij = A_ij * b_ij + a_i if i == j
        Let c_i = f(z_i) + sum_j A_ij g(z_i, z_j)

        d e_i / dt = Y @ e + c (Linear ODE system)
        ==== Background ====

        This function computes d e_i / dt
        '''
        Y = self.dFdx(x, 0, param)
        c = self.dxdt(x, 0, param)
        return Y @ err + c


    def perturb_err_matrix(self, param, x):
        Y = self.dFdx(x, 0, param)
        c = self.dxdt(x, 0, param)
        return Y, c

    def perturb_err_matrix_sp(self, param, x):
        A = self.A
        N = self.N
        R = param[0]
        B = 1 
        # Extract the row and column indices of the non-zero elements 
        adj_list = np.zeros((self.num_edges+N,2), dtype=np.int64)
        data = np.zeros(self.num_edges+N)
        adj_list[:self.num_edges,0], adj_list[:self.num_edges,1] = self.A.nonzero() 

        adj_list[self.num_edges:, 0] = list(range(N))
        adj_list[self.num_edges:, 1] = list(range(N)) 
         
        data[:self.num_edges] = (R * (1 - x[adj_list[:self.num_edges,0]])) 
        data[self.num_edges:] = - B - R * (A @ x) 
        J = sp.csc_matrix((data, (adj_list[:, 0], adj_list[:, 1])), shape=(N, N),  dtype=float)
        c = self.dxdt(x, 0, param) 
        return J, c
     

    def dxdt_mfa(self, x,t, param, degree):
        '''
        Return the N-D mean-field formula f(x, param) + degree * g(x, xeff, param).
        '''  
         
        self.nfe_mfa += 1
        xeff = self.xeff 
        R = param[0]
        B = 1        
        dxdt = - B * x + R * (1 - x) * degree * xeff   
        return dxdt  
    
    def dxdt_xeff(self, x,t, param, beta=None):
        '''
        Return the 1-D mean-field formula f(x, param) + beta * g(x, x, param).
        '''
        if beta is None:
            beta = self.beta
        R = param[0]
        B = 1 
        return - B * x + beta * R * (1 - x) * x
 
    def dxdt(self, x,t, param, A=None):
        '''
        Compute the full dynamics f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        self.nfe_dxdt += 1
        if A is None:
            A = self.A
        R = param[0]
        B = 1 
        dxdt = - B * x + R * (1 - x) * (A @ x) 
        return dxdt
 