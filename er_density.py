import numpy as np 
import time
from utils import * 
import os  
import pandas as pd   
import datetime  
from methods import *  
 
SOLVER = 'odeint'

def er_density(args): 
    '''  
    RQ: How does the error of step 1(mfa) change w.r.t density? 
    '''
    # setup logger
    now = datetime.datetime.now() 
    args.date = str(now)  
    args.save_dir = f"./results/{args.experiment}/{args.date}"
    log_path = f'{args.save_dir}/result_{args.avg_operator}_{args.dynamics}_{args.n}.log'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))

    # Create network
    t1 = time.time()
    net = create_net(args) 
    t2 = time.time() 
     
    logger.info(f'Average degree {np.mean(net.degree)}')
    logger.info(f"# Unique Degrees = {len(np.unique(net.degree))}")
    logger.info(f"Degree Range = {np.min(net.degree)}, {np.max(net.degree)}")
    logger.info(f"Time to create network {t2 - t1}")
    ##################### Load ground truth parameters 
    gt = read_param(args) 
    net.gt = gt    
    density = np.linspace(0.01,1,10)
    if args.avg_operator == 'avg':
        avg_operator = L_avg
    elif args.avg_operator == 'in':
        avg_operator = L_2in
    n = args.n  
    G = nx.fast_gnp_random_graph(n, 0.5, seed=args.seed) 
    dic = {
        'num_node': [],
        'density': [], 
        'phase1_state_err_avg': [],
        'phase1_state_err_std': [],
        'phase2_state_err_avg': [],
        'phase2_state_err_std': [],
        'full_derive_avg': [],
        'full_derive_std': [], 
        'phase1_derive_avg': [],
        'phase1_derive_std': [],
        'phase2_derive_avg': [],
        'phase2_derive_std': [],
        'true_steady_state_runtime_avg': [],
        'true_steady_state_runtime_std': [], 
        'phase1_runtime_avg': [],
        'phase1_runtime_std': [],
        'phase2_runtime_avg': [],
        'phase2_runtime_std': [],
    }
    for k in range(len(density)):
        p = density[k]
        logger.info(f"\nN = {n} p = {p}")
        # create network 
        phase1_state_err_li = []
        phase2_state_err_li = []

        full_deriv_li = [] 
        phase1_deriv_li = []
        phase2_deriv_li = []

        true_steady_state_runtime_li = [] 
        phase1_runtime_li = []
        phase2_runtime_li = []
        for r in range(5): 
            G = nx.fast_gnp_random_graph(n, p, seed=r) 
            # assign topology
            net.setTopology(nx.to_numpy_array(G, nodelist=range(n)))
            
            # True Steady State
            true_steady_state_runtime, residual, true, full_deriv = full(net, args, logger) 
            full_deriv_li.append(full_deriv)
            true_steady_state_runtime_li.append(true_steady_state_runtime)
           
            foo = prt
            phase1_runtime, phase1_state_err, phase1_deriv, phase2_runtime, residual, phase2_state_err, phase2_deriv = foo(net, args, logger, true, L=avg_operator)
            phase1_state_err_li.append(phase1_state_err)
            phase2_state_err_li.append(phase2_state_err)

            phase1_deriv_li.append(phase1_deriv)
            phase2_deriv_li.append(phase2_deriv)

            phase1_runtime_li.append(phase1_runtime)
            phase2_runtime_li.append(phase2_runtime)
             
        dic['num_node'].append(n)
        dic['density'].append(p) 
        dic['phase1_state_err_avg'].append(np.mean(phase1_state_err_li))
        dic['phase1_state_err_std'].append(np.std(phase1_state_err_li))
        dic['phase2_state_err_avg'].append(np.mean(phase2_state_err_li))
        dic['phase2_state_err_std'].append(np.std(phase2_state_err_li))

        dic['full_derive_avg'].append(np.mean(full_deriv_li))
        dic['full_derive_std'].append(np.std(full_deriv_li)) 
        dic['phase1_derive_avg'].append(np.mean(phase1_deriv_li))
        dic['phase1_derive_std'].append(np.std(phase1_deriv_li))
        dic['phase2_derive_avg'].append(np.mean(phase2_deriv_li))
        dic['phase2_derive_std'].append(np.std(phase2_deriv_li)) 

        dic['true_steady_state_runtime_avg'].append(np.mean(true_steady_state_runtime_li))
        dic['true_steady_state_runtime_std'].append(np.std(true_steady_state_runtime_li)) 
        dic['phase1_runtime_avg'].append(np.mean(phase1_runtime_li))
        dic['phase1_runtime_std'].append(np.std(phase1_runtime_li))
        dic['phase2_runtime_avg'].append(np.mean(phase2_runtime_li))
        dic['phase2_runtime_std'].append(np.std(phase2_runtime_li))

    df = pd.DataFrame(dic)
    df.to_csv(f'{args.save_dir}/result_{args.avg_operator}_{args.dynamics}_{args.n}.csv', index=False)

