import numpy as np
import random
import time
from utils import * 
import os 
import pandas as pd  
import datetime   
import scipy.sparse as sp 
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import gmres, bicgstab
from methods import rk4
  
def huge_net(args):
    '''
    Approximate steady state using pure mean-field approach plus some round of full ODEs.

    Compare three different methods:
        full integration using RK
        fixing neighbor at constant 
        perturbation, integrate error
    '''
    # setup logger 
    now = datetime.datetime.now()
    date = str(now) # str(uuid.uuid4()) 
    trail = f"{args.dynamics}_{args.data}"
    log_path = f"./results/{args.experiment}/{date}/{trail}/result_{args.seed}.log"
    if not os.path.exists(f"./results/{args.experiment}/{date}/{trail}"):
        os.makedirs(f"./results/{args.experiment}/{date}/{trail}")
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))

    # Create network
    t1 = time.time()
    net = create_net(args)
    logger.info(f'{time.ctime()}') 
    #########################################
    # assumption: file is sender -> receiver
    directed = True
    weights = None 
    if args.data == 'CC-Neuron_cci.tsv': 
        file_path = f'data/{args.data}'  # Replace with the actual file path
        # Initialize variables
        edges = []
        node_to_int = {}
        edge_count = 0
        node_id = 0  # Counter for assigning unique integers to nodes
        with open(file_path, "r") as file: 
            # Skip the header line
            next(file) 
            # Process each line to build the mapping and edge list
            for line in file:
                protein1, protein2 = line.strip().split()
                
                # Assign an integer ID to each unique node
                if protein1 not in node_to_int:
                    node_to_int[protein1] = node_id
                    node_id += 1
                if protein2 not in node_to_int:
                    node_to_int[protein2] = node_id
                    node_id += 1
                
                # Store edges as integers
                edges.append([node_to_int[protein1], node_to_int[protein2]])
                edge_count += 1 
        adj_list = np.array(edges)  
        N =  len(node_to_int)
        M = edge_count
    elif args.data in ['PP-Decagon_ppi.edges.csv', 'PP-Pathways_ppi.edges.csv']:  
        adj_list = np.loadtxt(f"data/{args.data}", delimiter=',')
        if np.min(adj_list) > 0:
            adj_list -= np.min(adj_list)
        N = len(np.unique(adj_list))
        M = adj_list.shape[0]
    elif args.data[:4] == "bio-" or args.data in ['soc-pokec-relationships.txt','com-friendster.ungraph.txt']:
        adj_list1 = np.loadtxt(f"data/{args.data}", delimiter='\t') 
        unique_ind = np.unique(adj_list1[:,:-1])
        remap = {}
        count = 0
        for i in unique_ind:
            remap[i] = count 
            count += 1 

        adj_list = np.zeros((adj_list1.shape[0],2))
        for i in range(adj_list.shape[0]):
            adj_list[i][0] = remap[adj_list1[i][0]]

        M = adj_list.shape[0]
        weights = adj_list1[:,-1] 
        N = count
    else:
        f = open(f"data/{args.data}", "r")
        for i in range(3):
            data = f.readline()
        N = int(data.split()[2])
        M = int(data.split()[4])
        del data
        f.close()
        adj_list = np.loadtxt(f"data/{args.data}", delimiter='\t', skiprows=4)
    logger.info(f'{time.ctime()}') 
    logger.info(f"N = {N}, M = {M}") 
    logger.info(f'adj_list.shape {adj_list.shape}')
    E = adj_list.shape[0] 
    if weights is None:
        weights = np.ones(E)  
    A = sp.csc_matrix((weights, (adj_list[:, 1], adj_list[:, 0])), shape=(N, N),  dtype=np.int64)
    net.setTopology(A, sparse=True, directed=directed) 
    np.savetxt(f"./results/{args.data}_degree.txt", net.degree)
     
    t2 = time.time()
    method = [5] # full, powell, mfa+ avg, mfa+ wavg, perturbation avg, perturbation wavg
    method_name = ["Full", "Root", "MFA+ avg" , "MFA+ wavg", "Perturbation avg", "Perturbation wavg"]
    logger.info(f'Average degree {np.mean(net.degree)}')

    logger.info(f"Time to create network {t2 - t1}") 
    ###################################################
    # Load ground truth parameters
    gt = read_param(args) 
    net.gt = gt 
    runtimes = []
    dxdts = [] 
    t = np.linspace(0,10,2) 
    ###################################################
     
    logger.info(f"# Unique Degrees = {len(np.unique(net.degree))}")
    logger.info(f"Degree Range = {np.min(net.degree)}, {np.max(net.degree)}")
     
    mfa_res_deriv = []
    mfa_res_rt = [] 
 
    t1 = time.time() 
    logger.info(f'\nStart computing xeff weighted {time.ctime()}')  
  
    beta = net.out_degree @ net.degree / net.degree.sum() 
    if args.dynamics == 'epi':
        x0 = 0.5
    else:
        x0 = 20
    net.xeff = net.get_xeff(net.gt, x0, beta) 
    logger.info(f"xeff {net.xeff}")
    logger.info(f'Finish computing xeff {time.ctime()}\n')
      
    xhat_uni = net.solve_ode(net.dxdt_mfa, np.full(len(net.deg_unique), net.xeff), t, net.gt, net.deg_unique) 
    xhat_wavg  = np.zeros(net.N)
    for i in range(len(xhat_uni)):
        xhat_wavg[net.degree == net.deg_unique[i]] = xhat_uni[i]  
    np.savetxt(f"./results/{args.experiment}/{date}/{trail}/wMFA.txt", xhat_wavg)
    t2 = time.time()
    mfa_time_wavg = t2 - t1 
    abs_deriv = np.mean(np.abs(net.dxdt(xhat_wavg, 0, net.gt)))
    logger.info(f"mfa time {mfa_time_wavg}") 
    logger.info(f"mfa abs deriv {abs_deriv}")
    logger.info(f"Avg mean-field state {np.mean(xhat_wavg)}")

    mfa_res_deriv.append(abs_deriv)
    mfa_res_rt.append(mfa_time_wavg)
    
    pd.DataFrame({  
        "Abs Deriv": mfa_res_deriv,
        "Runtime": mfa_res_rt
    }).to_csv(f"./results/{args.experiment}/{date}/{trail}/mfa_err_{args.seed}.csv", index=None)
    ########## Finish Compute Mean-field State ########### 
      
    print('before perturb', net.num_edges)
    t1 = time.time()        
    y3 = perturbation(net.perturb_err_matrix_sp, xhat_wavg, net.unload(net.gt), logger, max_iter=20, lambd=1e-4, h = args.h, net=net)
    t2 = time.time() 
    abs_deriv = np.mean(np.abs(net.dxdt(y3, 0, net.gt)))
    np.savetxt(f"./results/{args.experiment}/{date}/{trail}/SS_Method5.txt", y3)
    logger.info(f"Runtime of perturbation method {mfa_time_wavg+t2 - t1} ") 
    logger.info(f"deriv error  {abs_deriv}\n")
    logger.info(f"Avg steady states {np.mean(y3)}")
     
    #### Save results ####
    runtimes.append(mfa_time_wavg+t2-t1)
    dxdts.append(abs_deriv)  
    pd.DataFrame({
        "dataset": [args.data] * len(method),
        "Node Number": [N] * len(method),
        "Edge Number": [M] * len(method),
        "Method": [method_name[i] for i in method],
        "Runtime (s)": runtimes,
        "Average Absolute Derivative": dxdts,
    }).to_csv(f"./results/{args.experiment}/{date}/{trail}/runtime_{args.seed}.csv", index=None)
     
def perturbation(err_matrix, y0, param, logger, h=0.01, max_iter=100, lambd=1e-7, net=None ):
    '''
    Perturbation for sparse matrices.

    Let e_i = x_i - z_i 

    d e_i / dt ~ [f'(z_i) + sum_j A_ij g_1 (z_i, z_j)] e_i + sum_j A_ij g_2 (z_i, z_j) e_j + f(z_i) + sum_j A_ij g(z_i, z_j) 
    where g1 (g2) is the partial derivative of g w.r.t. the first (second) argument
    
    Let a_i = [f'(z_i) + sum_j A_ij g_1 (z_i, z_j)]
    Let b_ij = g_2 (z_i, z_j)
    Let B_ij = A_ij * b_ij if i != j; B_ij = A_ij * b_ij + a_i if i == j
    Let c_i = f(z_i) + sum_j A_ij g(z_i, z_j) 

    d e_i / dt = B @ e + c (Linear ODE system)
    To compute the equilibrium of this system, there are several ways to do this:
        1. Direct method: compute the analytical solution to this system and take t to infinity
            - Suppose B has real eigenvectors, the general solution of e_i is 
                e = sum_j c_j e^{lambda_j t} v_j
                where c_j are constants, lambda_j, v_j are the jth eigenvalue, eigenvector respectively.
                To determine c_j, we would need n pairs of points (t, e)
        2. Numerical method: RK4 from a small IC
        3. Solving a linear system B @ e = - c using a numerical method starting from a small e. 
    ''' 
    y = y0
    np.random.seed(0)
    err = np.random.normal(0,0.001,len(y)) 
    step = 0 
    prev_err = np.inf
    while np.mean(np.abs(err)) > 1e-8: 
        if step == max_iter:
            logger.info(f"perturbation ==> reach max iter")
            break 
        logger.info(f"Perturbation Step {step}") 
        B, c = net.perturb_err_matrix_sp(param, y)  
 
        logger.info(f"  Computed matrix {time.ctime()} nonzeros = {B.getnnz()}") 
         
        err, _ = bicgstab(B, -c, tol=1e-15)
        logger.info(f"  Computed error {time.ctime()}: {np.mean(np.abs(err))}")  
        y = y + err
        abs_deriv = np.mean(np.abs(net.dxdt(y, 0, net.gt)))
        logger.info(f"<|dxd|> {abs_deriv}")
        if abs_deriv > prev_err:
            logger.info(f"Break: Abs deriv increasing")
            break 
        prev_err = abs_deriv
        step += 1
    return y  