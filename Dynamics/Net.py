import numpy as np
from scipy.integrate import odeint  
import networkx as nx  

class Net(object):
    def __init__(self, N, p, dyn, name, topo='er', m=4, directed=False, seed=42):
        self.N = N
        self.name = name
        self.topo = topo
        self.dyn = dyn
        self.nfe_mfa = 0
        self.nfe_dxdt = 0
        if topo == 'er':
            G = nx.fast_gnp_random_graph(N, p, seed=seed)
        else:
            G = nx.barabasi_albert_graph(N, m, seed=seed) 
        self.setTopology(nx.to_numpy_array(G, nodelist=range(N)))
        self.int_step_size = 0.001
 
        self.count = 0
    def unload(self, param):
        '''
        Unload param, add extra parameters if needed.
        '''
        return param 
    def dxdt(self, x, t, param, degree):
        '''
        Compute the full dynamics f(x_i, param) + sum_j A_ij g(x_i, x_j, param) for all i.
        '''
        pass 
    def setTopology(self, A, A_exist=True, directed=False, sparse=False):
        if A_exist:
            if sparse:  
                self.A = A 
                self.degree = np.asarray(A.sum(1)).reshape(-1)
                self.out_degree = np.asarray(A.sum(0)).reshape(-1)
                self.num_edges = A.nonzero()[0].shape[0] 
                print('self.num_edges', self.num_edges) 
                self.deg_unique = np.unique(self.degree)
                self.N = A.shape[0] 
            else:
                self.A = A
                self.degree = self.A.sum(axis=1)
                self.out_degree = self.A.sum(0)
                binA = np.zeros((len(A),len(A)))
                binA[A != 0] = 1
                self.degree_k = binA.sum(axis=1) # to distinguish between degree and weighted degree
                self.deg_unique = np.unique(self.degree) 
                self.N = A.shape[0] 


        else:
            self.G = A  # A is a graph instance, instead of an adjacency matrix
            self.A = nx.to_numpy_array(self.G)
            self.N = len(self.G.nodes)
            self.degree = np.empty(self.N)
            self.deg_unique, self.block_sizes = np.unique(self.degree, return_counts=True)
            self.out_degree = np.empty(self.N)
            self.directed = directed
            for i in (self.G.nodes):
                if self.directed:
                    self.degree[i] = self.G.in_degree(i)
                    self.out_degree[i] = self.G.out_degree(i)
                else:
                    self.degree[i] = self.G.degree(i)
                    self.out_degree[i] = self.G.degree(i)
            self.beta = np.mean(self.out_degree * self.degree) / np.mean(self.degree)
            self.beta_star = self.Lstar(self.degree)
            self.H = np.std(self.out_degree) * np.std(self.degree) / np.mean(self.degree)   
        self.A_exist = A_exist
        self.obs_idx = list(range(self.N))
    
    def get_xeff(self, param, y, beta):
        '''
        Solve f(x, param) + beta * g(x, x, param) = 0 for x.
        '''
        t_final = 5
        nsteps = 1
        t = np.linspace(0,t_final,t_final*nsteps) 
        xeff = self.solve_ode(self.dxdt_xeff, np.array([np.mean(y)]), t, param, beta )
        return xeff
    

    def solve_ode(self, dxdt, x0, t, param, degree=None, h=0.001, jac=None, method=None, k=np.inf):
        '''
        Iteratively calls odeint to ensure correctness of steady states, i.e., dxdt close to zero.
        '''
        res = odeint(dxdt, y0=x0, t=t, args=(self.unload(param), degree))[-1]
        max_iter = 100
        itr = 0 
        while np.mean(np.abs(dxdt(res, t, self.unload(param), degree))) > 1e-9:  
            if itr == max_iter:
                break
            res = odeint(dxdt, y0=res, t=t, args=(self.unload(param), degree))[-1]
            itr += 1 
        return res
     