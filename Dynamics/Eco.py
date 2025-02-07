import numpy as np
from Dynamics.Net import Net 

class Eco(Net):
    '''
    Create a set of methods for Ecology dynamics.
    '''
    def __init__(self, N=10, p=3, name='', topo='er', m=4, seed=42):
        Net.__init__(self, N, p, 'eco', name, topo, m, seed=seed)
        self.gt = np.array([0.1, 5, 1, 5, 0.9, 0.1])
    
    def f(self, x, param=None):
        if param is None:
            param = self.gt
        B, K, C, D, E, H = self.unload(param)
        return B + x * (1 - x/K) * (x/C - 1)

    def g(self, x, y, param=None):
        if param is None:
            param = self.gt
        B, K, C, D, E, H = self.unload(param)
        return x * y / (D + E*x + H*y)

    def dFdx(self, x, t, param, A=None):
        '''
        Let a_i = [f'(z_i) + sum_j A_ij g_1 (z_i, z_j)]
        Let b_ij = g_2 (z_i, z_j)
        Let B_ij = A_ij * b_ij if i != j; B_ij = A_ij * b_ij + a_i if i == j 

        return B
        '''
        B,K,C,D,E,H = param 
        A = self.A 
        N = len(A)
        Z = np.repeat(x, N).reshape((N, N))

        G = D + E * Z + H * (Z.T)
        
        Y = A * Z / G - H * A * np.outer(x, x) / (G * G)

        Y1 = A * Z.T / G - E * A * np.outer(x, x) / (G * G)

        diag = (1 - x / K) * (x / C - 1) + x * (1 - x / C) / K + x * (1 - x / K) / C + \
                np.sum(Y1, axis=1)

        np.fill_diagonal(Y, diag)
        return Y

    def dFdx_mfa(self, x, t, param, degree=None): 
        B, K, C, D, E, H = param
        xeff = self.xeff
        F_x = 2 * x * (1 / C + 1 / K) - 3 * x * x / (K * C) - 1 +\
            degree * (xeff / (D + E * x + H * xeff) - \
                     x * xeff * E / ((D + E * x + H * xeff) ** 2))
        return np.diag(F_x)

    def dFdx_brn(self, x, t, param, degree=None):
        if len(param) == 5:
            B, K, C, D, E = param
        else:
            B, K, C, D, E, H = param
            E = E + H
        F_x = 2 * x * (1 / C + 1 / K) - 3 * x * x / (K * C) - 1 +\
            degree * (2 * x / (D + E * x) - x * x * E / ((D + E * x) ** 2))
        return np.diag(F_x)

    def dxdt_err(self, err, t, param, x):
        '''
        ==== Background ====
        Ref:  def perturbation()

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
    
    def dxdt_mfa(self, x,t, param, degree):
        '''
        Return the N-D mean-field formula f(x, param) + degree * g(x, xeff, param).
        ''' 
        self.nfe_mfa += 1 
        xeff = self.xeff   
        B,K,C,D,E,H = param
        return B + x * (1 - x/K) * (x/C - 1) + degree * (x*xeff) / (D + E*x + H*xeff)
     
    def dxdt_xeff(self, x,t, param, beta=None):
        '''
        Return the 1-D mean-field formula f(x, param) + beta * g(x, x, param).
        '''
        B,K,C,D,E,H = param
        return B + x * (1 - x/K) * (x/C - 1) + beta * (x ** 2) / (D + (E+H)*x)
          

    def dxdt(self, x, t, param, A=None):
        '''
        Compute the full dynamics f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        if A is None:
            A = self.A
        self.nfe_dxdt += 1
        N = self.N
        B,K,C,D,E,H = param
        X = np.repeat(x, N).reshape(N,N)
        M = (np.outer(x,x)) / (D + E * X + H * X.T)
        dxdt = B+x*(1-x/K)*(x/C-1)+ (A * M).sum(1)
        return dxdt
 