import numpy as np
from Dynamics.Net import Net
import scipy.sparse as sp

class Gene(Net):
    def __init__(self, N=10, p=3, name='', topo='er', m=4, seed=42):
        Net.__init__(self, N, p, 'gene', name, topo,m, seed=seed)
        self.gt = np.array([1,1,2])

    def f(self, x):
        B,f,h = self.gt
        return - B * (x ** f)
    def g(self, x, y, d=0):
        B,f,h = self.gt
        return (y ** h) / (y ** h + 1)

    def dFdx(self, x, t, param, A=None):
        A = self.A
        N = self.N
        B, f, h = param 
        g2 = h * (x ** (h-1)) / ((x ** h + 1) ** 2)
        Y = np.repeat(g2, N).reshape(N,N)
        Y = A * (Y.T)
        diag = - B * f * x ** (f - 1)
        np.fill_diagonal(Y, diag)
        return Y
 
 
    def dxdt_mfa(self, x, t, param, degree):
        '''
        Return the N-D mean-field formula f(x, param) + degree * g(x, xeff, param).
        '''
        #xeff = self.Lstar(x)
        xeff = self.xeff
        #xeff = np.mean(x)
        #xeff = self.xeff_coeff @ x
        B, f, h = param
        xh = xeff ** h 
        xf = x ** f 
        return - B * xf + degree * (xh / (xh + 1))
 
    def dxdt_xeff(self, x, t, param, beta=None):
        '''
        Return the 1-D mean-field formula f(x, param) + beta * g(x, x, param).
        '''
        if beta is None:
            beta = self.beta
        B, f, h = param
        xf = x ** f  
        xh = x ** h 

        dxdt = - B * xf + beta * (xh / (xh + 1))
        return dxdt
 
    def dxdt(self, x, t, param, A=None):
        '''
        Compute the full dynamics f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        if A is None:
            A = self.A
        B, f, h = param
        xf = x ** f 
        xh = x ** h
        dxdt = - B * xf + A @ (xh / (xh + 1))
        return dxdt 
         

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
        ''' 
        - Let F: R^N -> R^N denote the ODE
        - 'data' stores the nonzero entries of the Jacobian matrix of F at x. 
        - data[:self.num_edges] stores the off-diagonal entries and data[self.num_edges:] stores the diagonal entries. 
        - The off-diagonal entries are the derivative of F w.r.t. x_j
        - The diagonal entries are the derivative of F w.r.t. x_i
        '''
        A = self.A
        N = self.N
        B, f, h = param 
        # Extract the row and column indices of the non-zero elements
         
        adj_list = np.zeros((self.num_edges+N,2), dtype=np.int64)
        data = np.zeros(self.num_edges+N)
        adj_list[:self.num_edges,0], adj_list[:self.num_edges,1] = self.A.nonzero() 

        adj_list[self.num_edges:, 0] = list(range(N))
        adj_list[self.num_edges:, 1] = list(range(N)) 
          
        data[:self.num_edges] = h * (x[adj_list[:self.num_edges,0]] ** (h-1)) / ((x[adj_list[:self.num_edges,0]] ** h + 1) ** 2)
        data[self.num_edges:] = - f * B * (x ** (f-1))
        J = sp.csc_matrix((data, (adj_list[:, 0], adj_list[:, 1])), shape=(N, N),  dtype=float)
        c = self.dxdt(x, 0, param)
         
        return J, c
    
  