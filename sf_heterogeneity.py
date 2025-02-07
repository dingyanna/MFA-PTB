import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
from utils import *
import datetime
import calendar
from methods import *
import powerlaw
from tqdm import tqdm
import matplotlib.pyplot as plt 
 
 
def create_scale_free_network(gtype, num, gamma, min_degree=4):
    a = np.random.pareto(gamma-1, size=num) + 1
    print(np.min(a), np.max(a))
    degrees1 = np.round(a * min_degree).astype(int)


    if np.sum(degrees1) % 2 == 1:
        degrees1[0] += 1
    G = nx.configuration_model(degrees1)   
    selfloop_nodes = [i for (i,j) in nx.selfloop_edges(G)] 
    def remove_pair(node_id, G):
        j = selfloop_nodes[node_id]
        jp1 = selfloop_nodes[node_id+1]

        j_neighbor = [_ for _ in G.neighbors(j)]
        jp1_neighbor = [_ for _ in G.neighbors(jp1)]

        # remove two self edges
        G.remove_edge(j,j)
        G.remove_edge(jp1,jp1)

        # add an edge adjacent to j 
        if jp1 not in j_neighbor:
            G.add_edge(j,jp1)
        else:
            popu = [i for i in range(len(G)) if i not in j_neighbor]
            if len(popu) > 0: 
                idx = random.sample([i for i in range(len(G)) if i not in j_neighbor] ,1)[0]
                G.add_edge(j,idx)
        # add an edge adjacent to jp1 
        popu = [i for i in range(len(G)) if i not in jp1_neighbor ]
        if len(popu) > 0:
            idx = random.sample(popu,1)[0]
            G.add_edge(idx,jp1)
        return G

    if len(selfloop_nodes) % 2 == 1:
        for i in range(0,len(selfloop_nodes)-1,2):
            G = remove_pair(i, G)
        j = selfloop_nodes[-1]
        j_neighbor = [_ for _ in G.neighbors(j)]
        G.remove_edge(j,j)
        popu = [i for i in range(len(G)) if i != j_neighbor]
        if len(popu) > 0:
            idx = random.sample(popu,1)[0] 
            G.add_edge(selfloop_nodes[-1],idx) 
    else:
        for i in range(0,len(selfloop_nodes),2): 
            G = remove_pair(i, G)
    assert(len([i for (i,j) in nx.selfloop_edges(G)]) == 0)

    if gtype == 'gcc':
        # choose the biggest connected component
        G = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
   
    degree = list(G.degree)
    degree = [degree[i][1] for i in range(len(G.degree))]

    # print(degree)
    results = powerlaw.Fit(degree)
    gamma = results.alpha 
    return G 

def rewire(G):
    # Randomly remove an edge and add an non-existing one 
    while True:
        u = random.sample(range(len(G)), 1)[0]
        u1 = random.sample([i for i in range(len(G)) if i != u], 1)[0]
        neighbors = [i for i in G.neighbors(u)]
        if len(neighbors) > 1:
            break 
    
    v = random.sample(neighbors, 1)[0]
    u1_neighbor = [i for i in G.neighbors(u1)]
    u1_potential_neighbor = [i for i in range(len(G)) if i not in u1_neighbor]
    v1 = random.sample(u1_potential_neighbor, 1)[0]
    G.remove_edge(u,v)
    G.add_edge(u1,v1)
    return G 

def generate_networks(gtype, num):
    #########
    # Start with a network with gamma 3.5 
    # Rewire 10000 times, take the graph at every 1000 step to form a list of graphs with decreasing heterogeneity 
    #########
    G = create_scale_free_network(gtype, num, 3.5, min_degree=4) 
    heterogeneity = []
    rewire_time = []
    density = []
    G_lis = []
    for i in range(10000): 
        G = rewire(G)
        if i % 1000 == 0:
            rewire_time.append(i+1)
            degree = [d for n, d in G.degree()]
            het = np.std(degree)**2 / np.mean(degree)
            heterogeneity.append(het)
            density.append(np.mean(degree))
            G_lis.append(G.copy())
    return G_lis, heterogeneity

def sf_heterogeneity(args):
    # setup logger
    now = datetime.datetime.now()
    date = str(now)  
    args.save_dir = f"./results/{args.experiment}/{date}"
    log_path = f"{args.save_dir}/result_{args.dynamics}.log"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(f'{args.save_dir}/fig')
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))

    # Create network 
    if args.avg_operator == 'avg':
        avg_operator = L_avg
    elif args.avg_operator == 'in':
        avg_operator = L_2in 
    dic = {
        'num_node': [],
        'gamma': [],
        'heterogeneity': [],   
        'phase1_state_err_avg': [],
        'phase1_state_err_std': [],
        'phase2_state_err_avg': [],
        'phase2_state_err_std': [],
        'rk4_derive_avg': [],
        'rk4_derive_std': [], 
        'phase1_derive_avg': [],
        'phase1_derive_std': [],
        'phase2_derive_avg': [],
        'phase2_derive_std': [],
        'rk4_runtime_avg': [],
        'rk4_runtime_std': [], 
        'phase1_runtime_avg': [],
        'phase1_runtime_std': [],
        'phase2_runtime_avg': [],
        'phase2_runtime_std': [],
    }  
      
    phase1_state_err_li =np.zeros((5,10))
    phase2_state_err_li =np.zeros((5,10))
    full_deriv_li = np.zeros((5,10))
      
    phase1_deriv_li = np.zeros((5,10))
    phase2_deriv_li = np.zeros((5,10))
    true_steady_state_runtime_li = np.zeros((5,10))
     
    phase1_runtime_li = np.zeros((5,10))
    phase2_runtime_li = np.zeros((5,10))
    heterogeneity_li =np.zeros((5,10)) 
    n_li = np.zeros((5,10))
    gamma_true_li = np.zeros((5,10))
    for i in range(5):
        for j in range(10):  
            G = nx.read_graphml(f"sf_networks/G{j}_seed0_n{args.n}.graphml")
         
            degree = [d for n, d in G.degree()]
            degree = np.asarray(degree) 
            heterogeneity = np.std(degree) ** 2 / np.mean(degree)
             
            heterogeneity_li[i,j] = heterogeneity
            gamma_true = powerlaw.Fit(degree).alpha
            gamma_true_li[i,j] = gamma_true
              
            args.n = G.number_of_nodes() 
            n_li[i,j] = args.n
            net = create_net(args)
 
            net.setTopology(nx.to_numpy_array(G)) 
            # True Steady State
            true_steady_state_runtime, residual, true, full_deriv = full(net, args, logger) 
            full_deriv_li[i,j] = full_deriv
            true_steady_state_runtime_li[i,j] = (true_steady_state_runtime)
             
            phase1_runtime, phase1_state_err, phase1_deriv, phase2_runtime, residual, phase2_state_err, phase2_deriv = prt(net, args, logger, true, L=avg_operator)
            phase1_state_err_li[i,j] = (phase1_state_err)
            phase2_state_err_li[i,j] = (phase2_state_err)

            phase1_deriv_li[i,j] = (phase1_deriv)
            phase2_deriv_li[i,j] = (phase2_deriv)

            phase1_runtime_li[i,j] = (phase1_runtime)
            phase2_runtime_li[i,j] = (phase2_runtime)
              
             
    dic['num_node'] = np.mean(n_li,0)
    dic['gamma'] = np.mean(gamma_true_li,0)
    dic['heterogeneity'] = np.mean(heterogeneity_li,0)
    
    dic['phase1_state_err_avg'] = np.mean(phase1_state_err_li, 0)
    dic['phase1_state_err_std'] = np.std(phase1_state_err_li, 0)
    dic['phase2_state_err_avg'] = np.mean(phase2_state_err_li, 0)
    dic['phase2_state_err_std'] = np.std(phase2_state_err_li, 0)
    dic['rk4_derive_avg'] = np.mean(full_deriv_li, 0)
    dic['rk4_derive_std'] = np.std(full_deriv_li, 0)
    dic['phase1_derive_avg'] = np.mean(phase1_deriv_li, 0)
    dic['phase1_derive_std'] = np.std(phase1_deriv_li, 0)
    dic['phase2_derive_avg'] = np.mean(phase2_deriv_li, 0)
    dic['phase2_derive_std'] = np.std(phase2_deriv_li, 0)
    dic['rk4_runtime_avg'] = np.mean(true_steady_state_runtime_li, 0)
    dic['rk4_runtime_std'] = np.std(true_steady_state_runtime_li, 0)
     
    dic['phase1_runtime_avg'] = np.mean(phase1_runtime_li, 0)
    dic['phase1_runtime_std'] = np.std(phase1_runtime_li, 0)
    dic['phase2_runtime_avg'] = np.mean(phase2_runtime_li, 0)
    dic['phase2_runtime_std'] = np.std(phase2_runtime_li, 0)

    # check if all of the lengths are the same
    length = 0
    for idx, key in enumerate(dic.keys()):
        if idx == 0:
            length = len(dic[key])
            print('key', key, 'length:', length)
        else:
            if length != len(dic[key]):
                print('---error---')
                print(key)
                print(len(dic[key]))         
    df = pd.DataFrame(dic)
    store_path = f'{args.save_dir}/G_{args.avg_operator}_result_{args.dynamics}_{args.n}.csv'
    df.to_csv(store_path, index=False)